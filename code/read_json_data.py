import os
import json
import urllib.request

def retrieve_url(url, filename):
    if not os.path.exists(filename) and not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)
    else:
        print(f'{filename} exists! Nothing to download.')


def get_json_from_file(filename):
    with open(filename, 'r') as json_file:
        json_data = json.load(json_file)
    return json_data

def print_dict(json_dict, items=5):
    print({x: json_dict[x] for (i,x) in enumerate(json_dict) if i < items})


if __name__ == '__main__':
    # dataset https://www.ncdc.noaa.gov/cag/national/time-series/110/pcp/12/12/1895-2016
    url = 'https://www.ncdc.noaa.gov/cag/national/time-series/110-pcp-12-12-1895-2016.json'
    data_path = '../data'
    filename = f'{data_path}/climate.json'

    retrieve_url(url, filename)

    # Method 1 using Json load
    json_data = get_json_from_file(filename)
    print(f'Json Keys: {json_data.keys()}')
    print_dict(json_data['data'])

    # Method 2 using Json loads
    with open(filename, 'r') as json_file:
        raw_json = json_file.read()

    json_from_string = json.loads(raw_json)
    print(f'Json Keys: {json_from_string.keys()}')
    print_dict(json_from_string['data'])

    # clean data
    precip_raw = json_data['data']

    # method 1
    precipitation = [(x, float(precip_raw[x]['value'])) for x in precip_raw]
    print(precipitation[:5])

    # method 2
    precipitation = [(key, float(value['value'])) for key,value in precip_raw.items()]
    print(precipitation[:5])