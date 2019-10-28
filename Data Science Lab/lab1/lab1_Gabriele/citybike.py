import json
from math import cos, acos, sin


def load_data():
    with open("file.json") as f:
        obj = json.load(f)
    return obj['network']['stations']


def point2(stations_l):
    active_stations = [station for station in stations_l if station['extra']['status'] == "online"]
    print('Number of active stations', len(active_stations))


def point3(stations_l):
    """ I can do it with map() and filter()"""
    bikes_available = sum([station["free_bikes"] for station in stations_l])
    free_docks = sum([station["empty_slots"] for station in stations_l])
    print("Bikes available", bikes_available)
    print("Free docks", free_docks)


def distance_coords(lat1, lng1, lat2, lng2):
    """Compute the distance among two points."""
    deg2rad = lambda x: x * 3.141592 / 180
    lat1, lng1, lat2, lng2 = map(deg2rad, [ lat1, lng1, lat2, lng2 ])
    R = 6378100 # Radius of the Earth, in meters
    return R * acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))


def point4(stations_l, lat, lng):
    closest = (None, None)
    for station in stations_l:
        closest_station, closest_distance = closest
        current_distance = distance_coords(lat, lng, station["latitude"], station["longitude"])
        # if closest_distance is None, then we are at the first
        # loop execution where the station has available bikes.
        # In that case, we save the current station as the
        # closest one (as we do not have any other stations available).
        # From the next cycle on, to update `closest`, we need
        # the station to actually be closer than the already saved one.
        if station["free_bikes"] > 0 and (closest_distance is None or current_distance < closest_distance):
            closest = (station, current_distance)
    return closest


def main():
    stations_l = load_data()
    point2(stations_l)
    point3(stations_l)

    station, distance = point4(stations_l, 45.074512, 7.694419)
    print("Closest station:", station["name"])
    print("Distance:", distance, "meters")
    print("Number of available bikes:", station["free_bikes"])


if __name__ == '__main__':
    main()
