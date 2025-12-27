import requests
import math

class AmapLiveBridge:
    def __init__(self, api_key):
        self.key = api_key
        self.status_map = {"1": 1.0, "2": 1.8, "3": 4.0, "4": 12.0}

    def _convert_coords(self, lon, lat):
        """纠偏算法：OSM(WGS-84) -> 高德(GCJ-02)"""
        a, ee = 6378245.0, 0.006693421622965943
        dlat = self._transform(lon - 105.0, lat - 35.0, is_lat=True)
        dlng = self._transform(lon - 105.0, lat - 35.0, is_lat=False)
        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * math.sqrt(magic)) * math.pi)
        dlng = (dlng * 180.0) / (a / math.sqrt(magic) * math.cos(radlat) * math.pi)
        return lon + dlng, lat + dlat

    def get_live_weights(self, lon, lat, radius=1000):
        glon, glat = self._convert_coords(lon, lat)
        url = f"https://restapi.amap.com/v3/traffic/status/circle?key={self.key}&location={glon:.6f},{glat:.6f}&radius={radius}&extensions=all"
        try:
            roads = requests.get(url).json()['trafficinfo']['roads']
            return {r['name']: self.status_map.get(r['status'], 1.0) for r in roads}
        except: return {}

    def _transform(self, x, y, is_lat):
        if is_lat: return -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*x*y + 0.2*math.sqrt(abs(x))
        return 300.0 + x + 2.0*y + 0.1*x*x + 0.1*x*y + 0.1*math.sqrt(abs(x))