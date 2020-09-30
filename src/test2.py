#%%

from importlib import import_module


pkg = import_module("dlbd.models.CityNetTF2")
cls = getattr(pkg, "CityNetTF2")

print(cls)
