import json, sys

with open('/artefacts/assistant_bpe_tokenizer.json') as f:
    data = json.load(f)

assert 'model' in data, 'tokenizer JSON missing model key'
assert data.get('version'), 'tokenizer JSON missing version key'
print('tokenizer OK — vocab size:', len(data['model'].get('vocab', {})))
