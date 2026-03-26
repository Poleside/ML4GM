
'''
This code is used to extract meteorological variables from ERA 5 land data;
calculate the lapse rate based on the temperature information from  ERA5 pressure level data;
combine all the meteorological variables of each month of each glacier and then convert it to .csv file.

The part of extract meteorological variables from ERA 5 land data is inspired by the relevant section of
MassBalanceMachine(https://github.com/ODINN-SciML/MassBalanceMachine)

'''


import xarray as xr
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

climate_data = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/Result_evaluate/era5_data/ERA5_com_TS.nc'
geopotential_data = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/ERA5_data/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc'
t2_geo_pressure_level_data = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/Result_evaluate/era5_data/pressure_level_com_TS.nc'
glacier_data = '/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/predict_result/data_1959_pre/RGI_my_region_1950_pre.csv'

df = pd.read_csv(glacier_data)

with (xr.open_dataset(climate_data) as ds_c, \
        xr.open_dataset(geopotential_data) as ds_g, \
        xr.open_dataset(t2_geo_pressure_level_data) as ds_t2):

    ds_climate = ds_c.load()
    ds_geopotential = ds_g.load()
    ds_pressure_L = ds_t2.load()


    # Convert geopotential height to geometric height and add to dataset
    r_earth = 6367.47 * 10e3  # [m] (Grib1 radius)
    g = 9.80665  # [m/s^2]
    ds_geopotential_metric = ds_geopotential.assign(
        altitude_climate=lambda ds_geo: r_earth * ((ds_geopotential.z / g) / (r_earth - (ds_geopotential.z / g)))
    )

    # Get latitude and longitude
    lat = ds_climate.latitude
    lon = ds_climate.longitude

    # Data retrieved from: https://ecmwf-projects.github.io/copernicus-training-c3s/reanalysis-climatology.html
    # Adjust longitude coordinates so that the coordinates range from -180 to 180, instead of 0 to 360
    ds_180 = ds_geopotential_metric.assign_coords(longitude=(((ds_geopotential_metric.longitude + 180) % 360) - 180)).sortby('longitude')

    ds_geopotential_cropped = ds_180.sel(longitude=lon, latitude=lat, method='nearest')

    # Reduce expver dimension
    ds_climate = ds_climate.reduce(np.nansum, 'expver')
    ds_pressure_L = ds_pressure_L.reduce(np.nansum, 'expver')

    # Create list of climate name variables and months combined for one hydrological year
    climate_vars = list(ds_climate.keys())
    months_names = ['_oct', '_nov', '_dec', '_jan', '_feb', '_mar', '_apr', '_may', '_jun', '_jul', '_aug', '_sep']
    monthly_climate_vars = [f'{climate_var}{month_name:02}' for climate_var in climate_vars for month_name in
                            months_names]

    # Create list of lapse_rate variables and months combined for one hydrological year
    lapse_rate_vars = ['lapse_rate']
    monthly_lapse_rate_vars = [f'{lapse_rate_vars[0]}{month_name:02}' for lapse_rate_var in lapse_rate_vars for month_name in
                            months_names]

    # Initialize arrays for the climate variable per data point, altitude and lapse_rate
    climate_per_point = np.full((len(df), len(monthly_climate_vars)), np.nan)
    altitude_per_point = np.full((len(df), 1), np.nan)
    lapse_rate_per_point = np.full((len(df), len(monthly_lapse_rate_vars)), np.nan)

    stake_lat = df.CenLat.round(3)
    stake_lon = df.CenLon.round(3)
    #stake_date = pd.to_datetime(df['year'], format="%d/%m/%Y", errors='coerce')
    stake_year = df.year
    #stake_year = np.array(stake_date)

    # Iterate through stake data, and get the climate variables and altitude for this point
    for idx, (lat, lon, year) in enumerate(zip(stake_lat, stake_lon, stake_year)):

        # Some years are float NaNs, these cannot be processed and therefore will be skipped
        if math.isnan(year):
            continue

        range_date = pd.date_range(start=str(int(year) - 1) + '-09-01',
                                   end=str(int(year)) + '-09-01', freq='ME')

        # Select climate data and pressure level data for the point, or the nearest point to it in the range of the hydrological year
        climate_data_point = ds_climate.sel(latitude=lat, longitude=lon, time=range_date, method='nearest')
        pressure_L_data_point = ds_pressure_L.sel(latitude=lat, longitude=lon, time=range_date, method='nearest')
        #print(climate_data_point, pressure_L_data_point)

        # Convert selected data to Dataframe and save it
        if climate_data_point.dims:
            climate_points = climate_data_point.to_dataframe().drop(columns=['latitude', 'longitude'])
            climate_per_point[idx, :] = climate_points.to_numpy().flatten(order='F')

            # calculate lapse rate
            pressure_L_data_point = pressure_L_data_point.drop_sel(
                level=[1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250])
            time_pressure = pressure_L_data_point['time']
            df_rate_PL = pd.DataFrame(columns=['t', 'lapse_rate'])
            for time1 in time_pressure:
                pressure_L_data_point_level = pressure_L_data_point.loc[{'time': time1}]
                pressure_L_data_point_levels = pressure_L_data_point_level.to_dataframe().drop(columns=['latitude', 'longitude'])
                geo_height = r_earth * ((pressure_L_data_point_levels['z'] / g) / (r_earth - (pressure_L_data_point_levels['z'] / g)))
                rate_PL = np.polyfit(geo_height, pressure_L_data_point_levels['t'], 1)[0]
                df_rate_PL = df_rate_PL._append(pd.DataFrame({'t': [time1.values], 'lapse_rate': [rate_PL]}), ignore_index =True)
                #print(df_rate_PL)
            #print(df_rate_PL)

            # Select altitude data
            altitude_point = ds_geopotential_cropped.sel(latitude=lat, longitude=lon, method='nearest')
            altitude_per_point[idx] = altitude_point.altitude_climate.values[0]

            lapse_rate_per_point[idx,:] = df_rate_PL['lapse_rate']

    # Create DataFrames from arrays
    df_climate = pd.DataFrame(data=climate_per_point, columns=monthly_climate_vars)
    df_lapse_rate = pd.DataFrame(data=lapse_rate_per_point, columns=monthly_lapse_rate_vars)
    df_altitude = pd.DataFrame(data=altitude_per_point, columns=['altitude_climate'])

    # Concatenate DataFrames
    df_point_climate = pd.concat([df, df_climate, df_lapse_rate, df_altitude], axis=1)

    # Write to CSV
    #df_point_climate.to_csv(os.path.join(file_dir, file_name_out), index=False)
    df_point_climate.to_csv('/Users/yanfeipeng/Desktop/tu_graz/hma_mass_balance/geodetic_mb/test_based_BaranData/predict_result/data_1959_pre/predictX_1950_2023.csv')








