"""Download a coarse ERA5 pressure-level initial condition from CDS."""
import argparse
from pathlib import Path
import cdsapi
p=argparse.ArgumentParser();p.add_argument('--date',default='2025-03-06');p.add_argument('--hour',default='23:00');p.add_argument('--output',default='data/era5_20250306_23utc.nc');a=p.parse_args()
y,m,d=a.date.split('-');Path(a.output).parent.mkdir(parents=True,exist_ok=True)
request={'product_type':['reanalysis'],'variable':['geopotential','temperature','u_component_of_wind','v_component_of_wind','vertical_velocity'],'year':[y],'month':[m],'day':[d],'time':[a.hour],'pressure_level':['1','2','3','5','7','10','20','30','50','70','100','125','150','175','200','225','250','300','350','400','450','500','550','600','650','700','750','775','800','825','850','875','900','925','950','975','1000'],'grid':[2.5,2.5],'data_format':'netcdf','download_format':'unarchived'}
cdsapi.Client(quiet=True).retrieve('reanalysis-era5-pressure-levels',request,a.output)
print('Saved',a.output)
