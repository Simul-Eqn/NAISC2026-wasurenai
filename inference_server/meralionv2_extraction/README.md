## NOTE 
Need to monkeypatch meralion's code with ```modeling_bestrq_conformer.py``` (can find the cache dir where it errors to replace the file)

## making the venv 
I first installed torch with gpu acceleration in the conda env with pip install 
then I used a conda install for torchcodec[build=*cuda*] (as in environment.yml) 
that's all i remember.. 

## details 

For data structure in original: 
```
{
    'id': '00644',
    'path': '.cache/huggingface/datasets/downloads/extracted/f0b33b5266cd9452ee310eef3577cf7adb7f29aa54dbff74b9a8ee406a55d614/waves/ses2_spk3_L13101_189.900_5.490.wav',
    'audio': {
        'path': '.cache/huggingface/datasets/downloads/extracted/f0b33b5266cd9452ee310eef3577cf7adb7f29aa54dbff74b9a8ee406a55d614/waves/ses2_spk3_L13101_189.900_5.490.wav',
        'array': array([-6.1035156e-05, -1.8310547e-04, 3.0517578e-05, ...,
            0.0000000e+00, -3.0517578e-05, 0.0000000e+00
        ], dtype = float32),
        'sampling_rate': 16000
    },
    'transcription': '因为你不可能邀你的female friends去说走我们去play basketball',
    'duration': 5.489999771118164,
    'language': 'mixed',
    'original_speaker_id': 3,
    'session_id': 2,
    'topic': 'sports'
}
```


Timestamp is calculated by a cumulative sum of "duration" of each data point (indexed by id), starting from a set date/time (a veriable at the start of the program), plus "session_id" days. 

household_id is original_speaker_id 

sensor_id is "AUDIO" 

Just save a column for transcription as well 

sensor_id transitions are important but uncaptured... 


