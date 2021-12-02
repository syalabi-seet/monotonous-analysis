import os
import boto3
import time
import numpy as np
import argparse
import librosa
import warnings
import psycopg2
import tensorflow as tf
import soundfile as sf

from tensorflow.keras.layers import *
from tensorflow.keras import Model
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--access_id", dest="access_id", type=str)
parser.add_argument("--access_key", dest="access_key", type=str)
parser.add_argument("--host", dest="host", type=str)
parser.add_argument("--port", dest="port", type=str)
parser.add_argument("--user", dest="user", type=str)
parser.add_argument("--password", dest="password", type=str)
parser.add_argument("--dbname", dest="dbname", type=str)
   
parser.add_argument("--sample_rate", dest="sample_rate", type=int, default=16000)
parser.add_argument("--n_fft", dest="n_fft", type=int, default=2048)
parser.add_argument("--window_size", dest="window_size", type=int, default=400) # 25ms
parser.add_argument("--hop_length", dest="hop_length", type=int, default=160) # 10ms
parser.add_argument("--n_mels", dest="n_mels", type=int, default=64)
parser.add_argument("--n_mfcc", dest="n_mfcc", type=int, default=13)
parser.add_argument("--max_samples", dest="max_samples", type=int, default=64000)
parser.add_argument("--delta_width", dest="delta_width", type=int, default=3)
parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
parser.add_argument("--verbose", dest="verbose", type=int, default=1)

parser.add_argument("--mag_rate", dest="mag_rate", type=float, default=0.3)
parser.add_argument("--grad_limit", dest="grad_limit", type=float, default=0.2)
parser.add_argument("--benchmark_mul", dest="benchmark_mul", type=float, default=0.1)

# Save paths
parser.add_argument("--bucket_name", dest="bucket_name", type=str, default="audiosnippet-bucket")
parser.add_argument("--bucket_key", dest="bucket_key", default="outputs/snippets")
parser.add_argument("--local_dir", default="/opt/ml/processing")

args = parser.parse_known_args()[0]
seq_len = int(np.ceil(args.max_samples / args.hop_length))
input_shape = (seq_len, (args.n_mfcc * 3) + 2, 1)
parser.add_argument("--input_shape", type=tuple, default=input_shape)
parser.add_argument("--seq_len", type=int, default=seq_len)

args, _ = parser.parse_known_args()

class MonoTracker:
    def __init__(self, args):
        self.args = args
        self.segmentor = self.Segmentor(args)

    def Segmentor(self, args):
        spectrogram = Input(shape=args.input_shape, dtype=tf.float32, name='audio')
        mask = Input(shape=args.input_shape[0], dtype=tf.bool, name='mask')

        x = TimeDistributed(Conv1D(128, 3))(spectrogram)
        x = TimeDistributed(ReLU())(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPool1D(2))(x)

        x = TimeDistributed(Conv1D(64, 3))(x)
        x = TimeDistributed(ReLU())(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPool1D(2))(x)

        x = TimeDistributed(Conv1D(32, 3))(x)
        x = TimeDistributed(ReLU())(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPool1D(2))(x)
        
        x = TimeDistributed(Flatten())(x)
        x = Bidirectional(LSTM(100, dropout=args.dropout, return_sequences=True))(x, mask=mask)
        x = Bidirectional(LSTM(25, dropout=args.dropout, return_sequences=True))(x, mask=mask)
        x = TimeDistributed(Dense(2, activation='softmax'))(x, mask=mask)
        model = Model(inputs=[spectrogram, mask], outputs=x, name='Segmentor')
        model.load_weights(f"{self.args.local_dir}/models/segmentor.h5")
        return model

    def AudioPrep(self, y):
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.args.sample_rate, n_mfcc=self.args.n_mfcc, hop_length=self.args.hop_length,
            win_length=self.args.window_size, n_mels=self.args.n_mels,
            n_fft=self.args.n_fft, fmin=0, fmax=8000)[:, :self.args.seq_len]
        mfcc = np.transpose(mfcc)
        delta = librosa.feature.delta(
            mfcc, width=self.args.delta_width, order=1, axis=0)
        delta2 = librosa.feature.delta(
            mfcc, width=self.args.delta_width, order=2, axis=0)
        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=self.args.window_size, hop_length=self.args.hop_length)
        zcr =  np.transpose(zcr)[:-1, :]
        f0 = librosa.yin(y=y, sr=self.args.sample_rate, fmin=50, fmax=2000, win_length=self.args.window_size,
            hop_length=self.args.hop_length, frame_length=2048)
        f0 = np.expand_dims(np.diff(f0, axis=-1), axis=-1)
        mfcc = np.concatenate((mfcc, delta, delta2, zcr, f0), axis=-1)
        mfcc = tf.convert_to_tensor(mfcc, dtype=tf.float32)
        mfcc = tf.expand_dims(mfcc, axis=0)
        
        n_frames = len(y) // self.args.hop_length
        pad_length = self.args.seq_len - n_frames 
        mask = tf.concat([tf.ones([n_frames]), tf.zeros([pad_length])], axis=-1)
        mask = tf.cast(mask, dtype=tf.bool)
        mask = tf.expand_dims(mask, axis=0)    
        return {"audio": mfcc, "mask": mask}

    def MakeChunks(self, wav_path):
        # Resample
        y, sr = librosa.load(wav_path, sr=None)
        if sr != self.args.sample_rate:
            y = librosa.resample(
                y, orig_sr=sr, target_sr=self.args.sample_rate)
        chunk_length = self.args.hop_length * self.args.seq_len
        self.chunk_duration = chunk_length // self.args.sample_rate
        n_chunks = len(y) // chunk_length
        y = y[:(n_chunks * chunk_length)]
        y_chunks = np.array_split(y, n_chunks)
        return y_chunks

    def ChunkGenerator(self, wav_path):
        y_chunks = self.MakeChunks(wav_path)
        self.n_chunks = len(y_chunks)
        for y_chunk in y_chunks:
            if len(y_chunk) <= self.args.max_samples:     
                y_chunk = librosa.util.fix_length(
                    y_chunk, self.args.max_samples)
            yield y_chunk

    def predict(self, y_chunk):
        inputs = self.AudioPrep(y_chunk)
        y_pred = self.segmentor.predict(inputs)[0]
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.squeeze(np.where(y_pred==1))
        y_pred = [(y_pred[i].numpy(), y_pred[i+1].numpy()) for i, b in enumerate(y_pred) if i < len(y_pred) - 1]
        y_pred = [frames for frames in y_pred if ((frames[1] - frames[0]) >= 2) & ((frames[1] - frames[0] <= 10))]
        return y_pred
    
    def batch_predict(self, wav_path):
        false_counter, true_counter = 0, 0
        forward, residual, benchmark = 0, 0, 0
        total_mono, total_voiced, mean = 0, 0, 0
        logs = {
            'source_uri': wav_path,'segment_ratio': [], 'mono_segments': [], 
            'total_ratio': 0, 'output_uri': [], 'local_output_uri': [], 
            'start_time': [], 'end_time': [], 'frame_start': [],
            'frame_end': []}
        file_path = os.path.splitext(wav_path.split("/")[-1])[0]
        for i, y_chunk in enumerate(self.ChunkGenerator(wav_path)):
            total = 0
            start_time = time.time()
            y_pred = self.predict(y_chunk)
            if y_pred != [(2, 397)]:
                f0 = librosa.yin(
                    y=y_chunk, sr=self.args.sample_rate, fmin=50, fmax=2000,
                    win_length=self.args.window_size,
                    hop_length=self.args.hop_length, frame_length=2048)
                voiced_flag = self.get_voiced_flags(y_chunk, y_pred)
                voiced_sum = voiced_flag.sum()
            else: 
                voiced_sum = 0

            if voiced_sum == 0:
                if self.args.verbose == 1:
                    print("{}/{} - No speech detected.".format(i+1, self.n_chunks))
                false_counter += 1
                mono_ratio = 0
                pass
            else:
                words = self.extract_words(voiced_flag)
                for j, word in enumerate(words):
                    start_idx, end_idx = word
                    length = end_idx - start_idx
                    mean = np.mean(f0[start_idx:end_idx])
                    delta = np.abs(mean - residual)
                    if length > 1:
                        grad = np.abs(np.mean(np.gradient(f0[start_idx:end_idx], edge_order=1)))
                    else:
                        grad = 1
                    if forward == 0:
                        buffer = len(words) // 2
                    if ((forward == 0) & (j < buffer) | (delta > benchmark * self.args.benchmark_mul)):
                        benchmark += (mean / buffer)
                    else:
                        if (benchmark * self.args.mag_rate > delta) & (grad > self.args.grad_limit):
                            total += length
                    residual = mean
                forward = mean
                voiced_ratio = voiced_sum / len(voiced_flag)
                elapsed_time = time.time() - start_time
                mono_ratio = total / voiced_sum

                if mono_ratio >= 0.3:
                    if true_counter == 0:
                        seconds_start = i * self.chunk_duration
                        samples_start = librosa.time_to_samples(
                            seconds_start, sr=self.args.sample_rate)
                    else:
                        false_counter = 0
                    mono_block = True
                    true_counter += 1
                else:
                    mono_block = False
                    false_counter += 1
                    if (false_counter >= 1) & (true_counter > 0):
                        true_counter = 0
                        false_counter = 0

                        seconds_end = i * self.chunk_duration
                        samples_end = librosa.time_to_samples(
                            seconds_end, sr=self.args.sample_rate)
                        logs['start_time'].append(seconds_start)
                        logs['end_time'].append(seconds_end)
                        logs['mono_segments'].append([samples_start, samples_end])

                        ## Save y_chunk as .wav file
                        snippet_file = f"{file_path}_{seconds_start}-{seconds_end}.wav"
                        local_snippet_path = f"{self.args.local_dir}/snippets/{snippet_file}"
                        s3_snippet_path = f"s3://{self.args.bucket_name}/{self.args.bucket_key}/{snippet_file}"
                        logs['output_uri'].append(s3_snippet_path)
                        logs['local_output_uri'].append(local_snippet_path)

                if self.args.verbose == 1:
                    print("{}/{} - voiced_ratio: {:.2f}% - words: {} - monotonic_frames: {} - monotonic_ratio: {:.2f}% - elapsed_time: {:.2f}s".format(
                        i+1, self.n_chunks, voiced_ratio*100, len(words), total, mono_ratio*100, elapsed_time))
            
            logs['frame_start'].append(i*self.chunk_duration)
            logs['frame_end'].append(i*self.chunk_duration + self.chunk_duration)
            logs['segment_ratio'].append(mono_ratio)
            total_mono += total
            total_voiced += voiced_sum
        mono_ratio = (total_mono / total_voiced) * 100
        logs['total_ratio'] = mono_ratio
        return logs

    def extract_words(self, voiced_flag):
        word, words = [], []
        for i, j in enumerate(voiced_flag):
            if j == True:
                word.append(i)
            else:
                if len(word) > 1:
                    word = (word[0], word[-1])
                    words.append(word)
                word = []
        return words

    def get_voiced_flags(self, y, pred_frames):
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=args.hop_length)[0] 
        zcr = np.abs(np.gradient(zcr))
        zcr = np.less(zcr, zcr.mean())
        rms = librosa.feature.rms(y, hop_length=args.hop_length)[0] 
        rms = np.greater(rms, rms.mean())
        frames = np.zeros(len(zcr), dtype=int)
        for start_idx, end_idx in pred_frames:
            frames[start_idx:end_idx] = 1
        frames = np.logical_and(frames, zcr)
        if sum(frames) != 0:
            frames = np.logical_or(frames, rms)
        return frames

def save_segments(segments, snippet_paths):
    y, sr = librosa.load(wav_path, sr=None)
    if sr != args.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=args.sample_rate)
    for (samples_start, samples_end), snippet_path in zip(segments, snippet_paths):
        with sf.SoundFile(
                file=snippet_path, mode="w",
                samplerate=args.sample_rate, channels=1) as f:
            f.write(data=y[samples_start: samples_end])

if __name__ == '__main__':
    s3 = boto3.client('s3')   

    # Connect to RDS
    connection = psycopg2.connect(
        host=args.host,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
        connect_timeout=5)

    # Query RDS DB
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT input_uri, schedule_id, class_id " \
                "FROM class_recording " \
                "WHERE status = false")
            results = cursor.fetchall()
        print("INFO -- Input data extracted from class_recording.")

        model = MonoTracker(args)
        print("INFO -- Model initialized.")

        # Process files
        for s3_uri, schedule_id, class_id in results:
            filename = s3_uri.rsplit('/', 1)[-1]
            wav_key = s3_uri.split(f"s3://{args.bucket_name}/")[-1]
            wav_path = f"{args.local_dir}/inputs/{filename}"
            s3.download_file(args.bucket_name, wav_key, wav_path)
            print(f"INFO -- {filename} downloaded in local environment.")
            logs = model.batch_predict(wav_path=wav_path)

            # Save segments
            save_segments(
                segments=logs['mono_segments'], 
                snippet_paths=logs['local_output_uri'])
            print("INFO -- Segments splitted and saved locally.")  
        
            with connection.cursor() as cursor:
                # Update status in class_recording table
                cursor.execute(
                    "UPDATE class_recording " \
                    "SET status = true " \
                    "WHERE input_uri LIKE %s ESCAPE ''",
                    (s3_uri,))

                # Insert data in audio_analysis table
                for audio_id, (start_time, end_time, segment_ratio) in enumerate(
                        zip(logs['frame_start'], logs['frame_end'], logs['segment_ratio'])):
                    cursor.execute(
                        "INSERT INTO audio_analysis(audioanalysisid, start_time, end_time, monotone_percentage, class_id, schedule_id) " \
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (str(audio_id), str(start_time), str(end_time), str(segment_ratio), class_id, schedule_id)
                    )

                # Insert data in audio_clip_analysis table
                for start_time, end_time, output_uri in zip(
                    logs['start_time'], logs['end_time'], logs['output_uri']):
                    cursor.execute(
                        "INSERT INTO audio_clip_analysis(start_time, end_time, output_uri, monotone_percentage, class_id, schedule_id) " \
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (str(start_time), str(end_time), str(output_uri), logs['total_ratio'], class_id, schedule_id)
                    )

        connection.commit()
        print("INFO -- Database updated.")             

    print("INFO -- Job completed.")