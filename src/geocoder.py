import json
from shapely.geometry import shape, Point
import geopandas as gpd

DATA_PATH = '../data/apartments/renthop-nyc.csv'

import numpy as np
import pandas as pd

# Read New York City apartment rental listing data
df = pd.read_csv(DATA_PATH)
assert df.shape == (49352, 34)


# Remove the most extreme 1% prices,
# the most extreme .1% latitudes, &
# the most extreme .1% longitudes
df = df[(df['price'] >= np.percentile(df['price'], 0.5)) &
        (df['price'] <= np.percentile(df['price'], 99.5)) &
        (df['latitude'] >= np.percentile(df['latitude'], 0.05)) &
        (df['latitude'] < np.percentile(df['latitude'], 99.95)) &
        (df['longitude'] >= np.percentile(df['longitude'], 0.05)) &
        (df['longitude'] <= np.percentile(df['longitude'], 99.95))]


# load the geojson into geopandas
gdf = gpd.read_file('../data/apartments/nyc.geojson')


boro_name = []
nta_name = []

for i1, r1 in df.iterrows():
    p = Point(r1['longitude'], r1['latitude'])

    for i2, r2 in gdf.iterrows():
        poly = shape(r2['geometry'])

        if p.within(poly):
            boro_name.append(r2['boro_name'])
            nta_name.append(r2['ntaname'])

df['boro_name'] = boro_name
df['nta_name'] = nta_name


