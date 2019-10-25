import os
import glob
import netCDF4
import numpy as np
import datetime as dt
from itertools import cycle
from forest import geo
from forest import data
from forest import disk
import bokeh.models
import bokeh.palettes
from collections import defaultdict
from forest.util import initial_time

class ProfilesLoader(object):
    def __init__(self, paths):
        self.locator = ProfilesLocator(paths)

    @classmethod
    def from_pattern(cls, pattern):
        return cls(sorted(glob.glob(os.path.expanduser(pattern))))

    def profiles(self,
            initial_time,
            valid_time,
            variable,
            lon0,
            lat0):
        paths = self.locator.locate(initial_time)
        for path in paths:
            x, y = self.profiles_file(
                path,
                variable,
                valid_time,
                lon0,
                lat0)
            if x is not None and y is not None:
                data = {"x": list(x), "y": list(y)}
                break
        return data

    def profiles_file(self, *args, **kwargs):
        try:
            values, pressures = self._load_netcdf4(*args, **kwargs)
        except:
            values, pressures = self._load_cube(*args, **kwargs)
        return values, pressures

    def _load_cube(self, path, variable, valid_time, lon0, lat0):
        import iris
        cube = iris.load_cube(path, iris.Constraint(variable))
        
        times = cube.coord("time").points
        time_unit = cube.coord("time").units
        try:
            t = list(times).index(time_unit.date2num(valid_time))
        except:
            # Desired valid time is not in this file
            return None, None

        pressures = cube.coord('pressure').points
        lons = cube.coord('longitude').points
        lats = cube.coord('latitude').points
        if lons.ndim == 2:
            lons = lons[0, :]
        if lats.ndim == 2:
            lats = lats[:, 0]
        lons = geo.to_180(lons)
        i = np.argmin(np.abs(lons - lon0))
        j = np.argmin(np.abs(lats - lat0))

        values = cube.data

        # Index array based on coordinate ordering
        pts = []
        for c in cube.dim_coords:
            if c.name() == 'longitude':
                pts.append(i)
            elif c.name() == 'latitude':
                pts.append(j)
            elif c.name() == 'pressure':
                pts.append(slice(len(pressures)))
            elif c.name() == 'time':
                pts.append(t)
            else:
                pts.append(slice(None))
        pts = tuple(pts)
        values = values[pts]

        return values, pressures

    @staticmethod
    def _has_dim(cube, label):
        import iris
        try:
            cube.coord(label)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        return True

    def _load_netcdf4(self, path, variable, valid_time, lon0, lat0):
        with netCDF4.Dataset(path) as dataset:
            try:
                var = dataset.variables[variable]
            except KeyError:
                return [], []
            times = self._times(dataset, var)
            pts = self.search_times(times, valid_time)
            if not any(pts):
                # Desired valid_time was not in this file
                return None, None
            t = list(pts).index(True)
            lons = geo.to_180(self._longitudes(dataset, var))
            lats = self._latitudes(dataset, var)
            i = np.argmin(np.abs(lons - lon0))
            j = np.argmin(np.abs(lats - lat0))
            pressures = self._pressures(dataset, var)
            # Todo: This needs generalising to deal with data with an arbitrary number
            # of dimensions
            values = var[t, :, j, i]
           
        return values, pressures

    @staticmethod
    def _times(dataset, variable):
        """Find times related to variable in dataset"""
        time_dimension = variable.dimensions[0]
        coordinates = variable.coordinates.split()
        for c in coordinates:
            if c.startswith("time"):
                try:
                    var = dataset.variables[c]
                    return netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    pass
        for v, var in dataset.variables.items():
            if len(var.dimensions) != 1:
                continue
            if v.startswith("time"):
                d = var.dimensions[0]
                if d == time_dimension:
                    return netCDF4.num2date(var[:], units=var.units)

    def _pressures(self, dataset, variable):
        return self._dimension("pressure", dataset, variable)

    def _longitudes(self, dataset, variable):
        return self._dimension("longitude", dataset, variable)

    def _latitudes(self, dataset, variable):
        return self._dimension("latitude", dataset, variable)

    @staticmethod
    def _dimension(prefix, dataset, variable):
        for d in variable.dimensions:
            if not d.startswith(prefix):
                continue
            if d in dataset.variables:
                return dataset.variables[d][:]
        for c in variable.coordinates.split():
            if not c.startswith(prefix):
                continue
            if c in dataset.variables:
                return dataset.variables[c][:]

    @staticmethod
    def search(pressures, pressure, rtol=0.01):
        return np.abs(pressures - pressure) < (rtol * pressure)


    @staticmethod
    def search_times(times, time):
        return (times==time)


class ProfilesLocator(object):
    """Helper to find files related to Profiles"""
    def __init__(self, paths):
        self.paths = paths
        self.table = defaultdict(list)
        for path in paths:
            time = initial_time(path)
            if time is None:
                try:
                    with netCDF4.Dataset(path) as dataset:
                        var = dataset.variables["forecast_reference_time"]
                        time = netCDF4.num2date(var[:], units=var.units)
                except KeyError:
                    continue
            self.table[self.key(time)].append(path)
    
    def initial_times(self):
        return np.array(list(self.table.keys()),
                dtype='datetime64[s]')

    def locate(self, initial_time):
        if isinstance(initial_time, str):
            return self.table[initial_time]
        if isinstance(initial_time, np.datetime64):
            initial_time = initial_time.astype(dt.datetime)
        return self.table[self.key(initial_time)]

    __getitem__ = locate

    def key(self, time):
        return "{:%Y-%m-%d %H:%M:%S}".format(time)


class Profiles(object):
    def __init__(self, figure, loaders):
        self.figure = figure
        self.loaders = loaders
        self.sources = {}
        circles = []
        items = []
        colors = cycle(bokeh.palettes.Colorblind[6][::-1])
        for name in self.loaders.keys():
            source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
            })
            color = next(colors)
            r = self.figure.line(
                    x="x",
                    y="y",
                    color=color,
                    line_width=1.5,
                    source=source)
            r.nonselection_glyph = bokeh.models.Line(
                    line_width=1.5,
                    line_color=color)
            c = self.figure.circle(
                    x="x",
                    y="y",
                    color=color,
                    source=source)
            c.selection_glyph = bokeh.models.Circle(
                    fill_color="red")
            c.nonselection_glyph = bokeh.models.Circle(
                    fill_color=color,
                    fill_alpha=0.5,
                    line_alpha=0)
            circles.append(c)
            items.append((name, [r]))
            self.sources[name] = source

        self.figure.y_range.flipped = True

        legend = bokeh.models.Legend(items=items,
                orientation="horizontal",
                click_policy="hide")
        self.figure.add_layout(legend, "below")

        tool = bokeh.models.HoverTool(
                tooltips=[
                    ('Time', '@x{%F %H:%M}'),
                    ('Value', '@y')
                ],
                formatters={
                    'x': 'datetime'
                })
        self.figure.add_tools(tool)

        tool = bokeh.models.TapTool(
                renderers=circles)
        self.figure.add_tools(tool)

        # Underlying state
        self.state = {}

    @classmethod
    def from_groups(cls, figure, groups, directory=None):
        loaders = {}
        for group in groups:
            if group.file_type == "unified_model":
                if directory is None:
                    pattern = group.full_pattern
                else:
                    pattern = os.path.join(directory, group.full_pattern)
                loaders[group.label] = ProfilesLoader.from_pattern(pattern)
        return cls(figure, loaders)

    def on_state(self, app_state):
        next_state = dict(self.state)
        attrs = [
                "initial_time",
                "valid_time",
                "variable"]
        for attr in attrs:
            if getattr(app_state, attr) is not None:
                next_state[attr] = getattr(app_state, attr)
        state_change = any(
                next_state.get(k, None) != self.state.get(k, None)
                for k in attrs)
        if state_change:
            self.render()
        self.state = next_state

    def on_tap(self, event):
        self.state["x"] = event.x
        self.state["y"] = event.y
        self.render()

    def render(self):
        for attr in ["x", "y", "variable", "initial_time", "valid_time"]:
            if attr not in self.state:
                return
        x = self.state["x"]
        y = self.state["y"]
        variable = self.state["variable"]
        initial_time = dt.datetime.strptime(
                self.state["initial_time"],
                "%Y-%m-%d %H:%M:%S")
        valid_time = dt.datetime.strptime(
                self.state["valid_time"],
                "%Y-%m-%d %H:%M:%S")
        self.figure.title.text = variable
        for name, source in self.sources.items():
            loader = self.loaders[name]
            lon, lat = geo.plate_carree(x, y)
            lon, lat = lon[0], lat[0]  # Map to scalar
            source.data = loader.profiles(
                    initial_time,
                    valid_time,
                    variable,
                    lon,
                    lat)
