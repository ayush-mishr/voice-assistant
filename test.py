import json
d=json.JSONDecoder()
buf='{"a":1}{"b":2}'
print(d.raw_decode(buf))
