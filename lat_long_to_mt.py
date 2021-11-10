from geopy import distance

coords_1 = (52.2296756, 21.0122287)
coords_2 = (52.406374, 16.9251681)

print(distance.vincenty(coords_1, coords_2).km)
