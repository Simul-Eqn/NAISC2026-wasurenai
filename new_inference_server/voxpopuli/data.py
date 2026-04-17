from datasets import load_dataset

voxpopuli = load_dataset("facebook/voxpopuli", "en")

'''
{
  'audio_id': '20180206-0900-PLENARY-15-hr_20180206-16:10:06_5',
  'language': 11,  # "hr"
  'audio': {
    'path': '/home/polina/.cache/huggingface/datasets/downloads/extracted/44aedc80bb053f67f957a5f68e23509e9b181cc9e30c8030f110daaedf9c510e/train_part_0/20180206-0900-PLENARY-15-hr_20180206-16:10:06_5.wav',
    'array': array([-0.01434326, -0.01055908,  0.00106812, ...,  0.00646973], dtype=float32),
    'sampling_rate': 16000
  },
  'raw_text': '',
  'normalized_text': 'poast genitalnog sakaenja ena u europi tek je jedna od manifestacija takve tetne politike.',
  'gender': 'female',
  'speaker_id': '119431',
  'is_gold_transcript': True,
  'accent': 'None'
}
'''
