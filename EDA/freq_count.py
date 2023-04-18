import json

with open("pairs.txt", 'r') as f:
    count = {}
    exception = [str('"' + "o'clock" + '"')]
    for line in f:
        for e in exception:
            if e in line:
                new_e = e.replace("'", "").replace('"', "'")
                line = str(line).replace(e, new_e)
        line = json.loads(str(line).replace("'", '"'))
        for key in line:
            if key not in count:
                count[key] = 1
            else:
                count[key] += 1
    count = dict(sorted(count.items(), key=lambda x:x[1], reverse=True))
    with open("lost_count.json", "w") as output:
        json.dump(count, output)
