import matplotlib.pyplot as plt
import csv
from random import gauss
import datetime


def main():
    data = {
        'Date': [],
        'AverageTemperature': [],
        'AverageTemperatureUncertainty': [],
        'City': [],
        'Country': [],
        'Latitude': [],
        'Longitude': []
    }

    with open('GLT_filtered.csv') as f:
        skip = False
        for cols in csv.reader(f):
            if not skip:
                skip = True
                continue

            data['Date'].append(cols[0])
            data['AverageTemperature'].append(cols[1])
            data['AverageTemperatureUncertainty'].append(cols[2])
            data['City'].append(cols[3])
            data['Country'].append(cols[4])
            data['Latitude'].append(cols[5])
            data['Longitude'].append(cols[6])

    # print(data["Longitude"][:10])

    sort_by_date(data)
    for city in unique(data['City']):
        done = []
        print(city)
        if city not in done:
            done.append(city)
            indexes = value_filter(data['City'], city)

            for i in indexes:
                if data["AverageTemperature"][i] == '':
                    prev, next = get_neighbours(i, data['Date'], data['AverageTemperature'], indexes)
                    prev_val = float(data["AverageTemperature"][prev]) if prev != -1 else 0
                    next_val = float(data["AverageTemperature"][next]) if next != -1 else 0
                    new_val = (prev_val + next_val) / 2
                    data["AverageTemperature"][i] = new_val

            pass
    pass

    print(topN('Bangkok', 10, data))
    print(plot_distrib('Rome', 'Bangkok', data))


def value_filter(list, value):
    indexes = []
    for n, row in enumerate(list):
        if row == value:
            indexes.append(n)
    return indexes


def get_neighbours(n, date, avgs, indexes):
    prev_index = -1
    next_index = -1

    for i in indexes:
        if date[i] < date[n] or avgs[i] == '':
            continue
        next_index = i
        break

    for i in reversed(indexes):
        if date[i] > date[n] or avgs[i] == '':
            continue
        prev_index = i
        break

    return prev_index, next_index


def unique(list):
    res = []
    for item in list:
        if item not in res:
            res.append(item)
    return res


def topN(city, N, data, type='hot'):
    if N < 1:
        return []

    indexes = value_filter(data['City'], city)
    avgs = [float(data['AverageTemperature'][i]) for i in indexes]
    if type == 'hot':
        avgs.sort(reverse=True)
    else:
        if type == "cold":
            avgs.sort()

    return avgs[:N]


def plot_distrib(city1, city2, data):
    indexes1 = value_filter(data['City'], city1)
    values1 = [float(data['AverageTemperature'][i]) for i in indexes1]

    indexes2 = value_filter(data['City'], city2)
    values2 = [far_to_cels(float(data['AverageTemperature'][i])) for i in indexes2]
    '''
    avgs = [0] * 12
    num = [0]*12

    for i in indexes:
        date = datetime.datetime.strptime(data['Date'][i], "%Y-%m-%d")
        month = date.month

        temp = data['AverageTemperature'][i]
        avgs[month-1] += float(temp)
        num[month-1] += 1

    for i in range(12):
        avgs[i] = avgs[i] / num[i]
    '''

    plt.hist(values1)
    plt.hist(values2)
    plt.title(city1 + ' vs ' + city2)
    plt.show()

    # return avgs


def sort_by_date(data):
    to_sort = []

    for i in range(len(data['Date'])):
        to_sort.append((data['Date'][i], i))

    to_sort.sort(key=lambda x: x[0])

    sorted_indexes = [to_sort[j][1] for j in range(len(to_sort))]

    data['Date'] = [data['Date'][i] for i in sorted_indexes]
    data['AverageTemperature'] = [data['AverageTemperature'][i] for i in sorted_indexes]
    data['AverageTemperatureUncertainty'] = [data['AverageTemperatureUncertainty'][i] for i in sorted_indexes]
    data['City'] = [data['City'][i] for i in sorted_indexes]
    data['Country'] = [data['Country'][i] for i in sorted_indexes]
    data['Latitude'] = [data['Latitude'][i] for i in sorted_indexes]
    data['Longitude'] = [data['Longitude'][i] for i in sorted_indexes]


def far_to_cels(temp):
    return (float(temp) - 32) / 1.8


if __name__ == "__main__":
    main()
