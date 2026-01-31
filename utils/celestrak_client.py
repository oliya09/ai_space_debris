import requests
import json
import numpy as np
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ijson  # We'll use ijson for streaming JSON parsing

@dataclass
class TLEData:
    """Two-Line Element data structure."""
    norad_id: str
    name: str
    line1: str
    line2: str

class CelesTrakClient:
    """Client for fetching real-time satellite and debris data from CelesTrak."""
    
    BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
    
    def __init__(self):
        """Initialize CelesTrak client."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SpaceDebrisDashboard/1.0 (Educational)',
            'Accept': 'application/json'
        })
    
    def fetch_active_satellites(self, format_type: str = "json") -> List[Dict[str, Any]]:
        """
        Fetch all active satellites and debris from CelesTrak.
        
        Args:
            format_type: Data format ('json', 'tle', 'xml', 'csv')
            
        Returns:
            List of satellite/debris objects
        """
        try:
            print("🛰️ Fetching active satellites from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=active&FORMAT={format_type}"
            
            if format_type.lower() == "json":
                # For JSON, use streaming to handle large responses
                return self._fetch_json_streaming(url)
            else:
                # For TLE format
                return self._fetch_tle_format(url)
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching CelesTrak data: {str(e)}")
            raise
    
    def _fetch_json_streaming(self, url: str) -> List[Dict[str, Any]]:
        """Fetch JSON data using streaming to handle large responses."""
        try:
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type.lower():
                print(f"⚠️ Warning: Expected JSON but got {content_type}")
                print("🔄 Falling back to TLE format...")
                return self._fetch_tle_format(url.replace("FORMAT=json", "FORMAT=tle"))
            
            # Try to parse as regular JSON first
            try:
                data = response.json()
                print(f"✅ Retrieved {len(data)} active objects from CelesTrak")
                return data
            except json.JSONDecodeError:
                # If regular parsing fails, try streaming parse
                print("📡 Using streaming JSON parser for large dataset...")
                return self._stream_parse_json(response)
                
        except Exception as e:
            print(f"❌ Error in JSON streaming: {str(e)}")
            raise
    
    def _stream_parse_json(self, response) -> List[Dict[str, Any]]:
        """Parse large JSON response using streaming."""
        data = []
        buffer = ""
        brace_count = 0
        in_object = False
        
        for chunk in response.iter_content(decode_unicode=True, chunk_size=8192):
            if not chunk:
                continue
                
            buffer += chunk
            
            # Process buffer to extract JSON objects
            i = 0
            while i < len(buffer):
                char = buffer[i]
                
                if char == '[':
                    # Skip opening array bracket
                    i += 1
                    continue
                elif char == '{':
                    if not in_object:
                        start_idx = i
                    in_object = True
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and in_object:
                        # End of object
                        try:
                            obj_str = buffer[start_idx:i+1]
                            obj = json.loads(obj_str)
                            data.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping malformed object: {e}")
                        
                        in_object = False
                        # Remove processed part from buffer
                        buffer = buffer[i+1:]
                        i = 0
                        continue
                elif char == ']' and not in_object:
                    # End of array
                    break
                
                i += 1
            
            # Keep last partial object in buffer
            if in_object and len(buffer) > 100000:  # Prevent buffer from getting too large
                # Find last complete object
                last_brace = buffer.rfind('}')
                if last_brace != -1:
                    try:
                        obj_str = buffer[:last_brace+1]
                        obj = json.loads(obj_str[obj_str.find('{'):])
                        data.append(obj)
                    except:
                        pass
                    buffer = buffer[last_brace+1:]
        
        print(f"✅ Stream parsed {len(data)} objects")
        return data
    
    def _fetch_tle_format(self, url: str) -> List[Dict[str, Any]]:
        """Fetch and parse TLE format data."""
        try:
            response = self.session.get(url, timeout=90)
            response.raise_for_status()
            
            tle_data = self._parse_tle_data(response.text)
            print(f"✅ Retrieved {len(tle_data)} TLE objects from CelesTrak")
            return [self._tle_to_dict(tle) for tle in tle_data]
            
        except Exception as e:
            print(f"❌ Error fetching TLE data: {str(e)}")
            return []
    
    def fetch_debris_only(self) -> List[Dict[str, Any]]:
        """Fetch space debris specifically from active satellites data."""
        try:
            print("🗑️ Fetching space debris from CelesTrak...")
            
            # Get all active satellites and filter for debris
            active_data = self.fetch_active_satellites()
            
            # Filter for debris objects based on object names
            debris_objects = []
            for obj in active_data:
                name = obj.get('OBJECT_NAME', '').upper()
                if any(keyword in name for keyword in ['DEBRIS', 'DEB', 'FRAGMENT', 'FRAG']):
                    debris_objects.append(obj)
            
            print(f"✅ Retrieved {len(debris_objects)} debris objects from active satellites")
            return debris_objects
            
        except Exception as e:
            print(f"❌ Error fetching debris data: {str(e)}")
            return []
    
    def fetch_starlink_satellites(self) -> List[Dict[str, Any]]:
        """Fetch Starlink constellation satellites."""
        try:
            print("🌌 Fetching Starlink satellites from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=starlink&FORMAT=tle"
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            tle_data = self._parse_tle_data(response.text)
            print(f"✅ Retrieved {len(tle_data)} Starlink satellites as TLE")
            return [self._tle_to_dict(tle) for tle in tle_data]
            
        except Exception as e:
            print(f"❌ Error fetching Starlink data: {str(e)}")
            return []
    
    def fetch_recent_launches(self) -> List[Dict[str, Any]]:
        """Fetch recently launched objects (last 30 days)."""
        try:
            print("🚀 Fetching recent launches from CelesTrak...")
            
            url = f"{self.BASE_URL}?GROUP=last-30-days&FORMAT=tle"
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            tle_data = self._parse_tle_data(response.text)
            print(f"✅ Retrieved {len(tle_data)} recent launches as TLE")
            return [self._tle_to_dict(tle) for tle in tle_data]
            
        except Exception as e:
            print(f"❌ Error fetching recent launches: {str(e)}")
            return []
    
    def _parse_tle_data(self, tle_text: str) -> List[TLEData]:
        """Parse TLE format data."""
        lines = tle_text.strip().split('\n')
        tle_objects = []
        
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()
                
                # Extract NORAD ID from line 1
                norad_id = line1[2:7].strip()
                
                # Skip invalid TLEs
                if len(line1) >= 69 and len(line2) >= 69:
                    tle_objects.append(TLEData(norad_id, name, line1, line2))
        
        return tle_objects
    
    def _tle_to_dict(self, tle: TLEData) -> Dict[str, Any]:
        """Convert TLE data to dictionary format with orbital parameters."""
        try:
            # Parse orbital elements from TLE lines
            line1 = tle.line1
            line2 = tle.line2
            
            # Extract orbital elements from TLE
            inclination = float(line2[8:16])
            eccentricity = float("0." + line2[26:33])
            mean_motion = float(line2[52:63])
            mean_anomaly = float(line2[43:51])
            
            return {
                'NORAD_CAT_ID': tle.norad_id,
                'OBJECT_NAME': tle.name,
                'TLE_LINE1': line1,
                'TLE_LINE2': line2,
                'INCLINATION': inclination,
                'ECCENTRICITY': eccentricity,
                'MEAN_MOTION': mean_motion,
                'MEAN_ANOMALY': mean_anomaly
            }
        except (ValueError, IndexError) as e:
            print(f"⚠️ Error parsing TLE for {tle.norad_id}: {e}")
            return {
                'NORAD_CAT_ID': tle.norad_id,
                'OBJECT_NAME': tle.name,
                'TLE_LINE1': tle.line1,
                'TLE_LINE2': tle.line2,
                'INCLINATION': 45.0,
                'ECCENTRICITY': 0.001,
                'MEAN_MOTION': 15.0,
                'MEAN_ANOMALY': 0.0
            }
    
    def transform_to_dashboard_format(self, satellite_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform CelesTrak data to match our dashboard's expected format.
        
        Args:
            satellite_data: Raw satellite data from CelesTrak
            
        Returns:
            List of transformed objects ready for the dashboard
        """
        transformed_objects = []
        
        # Limit the number of objects to process for testing
        max_objects = 1000  # Adjust based on your needs
        if len(satellite_data) > max_objects:
            print(f"⚠️ Limiting to {max_objects} objects for processing (from {len(satellite_data)})")
            satellite_data = satellite_data[:max_objects]
        
        for obj in satellite_data:
            try:
                # Extract basic info
                norad_id = str(obj.get('NORAD_CAT_ID', 'UNKNOWN'))
                object_name = obj.get('OBJECT_NAME', 'UNKNOWN').strip()
                
                # Get orbital elements
                if 'MEAN_MOTION' in obj:
                    # JSON format
                    mean_motion = float(obj.get('MEAN_MOTION', 15.0))
                    inclination = float(obj.get('INCLINATION', 45.0))
                    eccentricity = float(obj.get('ECCENTRICITY', 0.001))
                else:
                    # TLE format or fallback values
                    mean_motion = 15.0  # Default for LEO
                    inclination = 45.0
                    eccentricity = 0.001
                
                # Calculate orbital period (minutes)
                if mean_motion > 0:
                    period = 1440.0 / mean_motion
                else:
                    period = 90.0  # Default LEO period
                
                # Simplified position calculation
                # For visualization purposes only - not accurate ephemeris
                altitude = self._estimate_altitude_from_period(period)
                
                # Calculate current position (simplified)
                current_time = datetime.now().timestamp()
                lat, lon = self._calculate_simplified_position(
                    inclination, period, current_time
                )
                
                # Calculate 3D coordinates
                earth_radius = 6371  # km
                total_radius = earth_radius + altitude
                x, y, z = self._calculate_3d_coordinates(lat, lon, total_radius)
                
                # Calculate orbital velocity
                mu = 398600.4418  # km³/s² (Earth's gravitational parameter)
                velocity = math.sqrt(mu / total_radius)  # km/s
                
                # Determine object characteristics
                object_type = self._classify_object(object_name, norad_id)
                size = self._estimate_size(object_type, object_name)
                risk_score = self._calculate_risk_score(altitude, size, object_type)
                
                # Create transformed object
                transformed_obj = {
                    'id': f"CT-{norad_id}",
                    'norad_id': norad_id,
                    'object_name': object_name,
                    'object_type': object_type,
                    'altitude': float(altitude),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'size': float(size),
                    'velocity': float(velocity),
                    'inclination': float(inclination),
                    'eccentricity': float(eccentricity),
                    'period': float(period),
                    'mean_motion': float(mean_motion),
                    'risk_score': float(risk_score),
                    'last_updated': datetime.now().isoformat(),
                    'data_source': 'CelesTrak',
                    'confidence': 0.95 if 'TLE_LINE1' not in obj else 0.85
                }
                
                transformed_objects.append(transformed_obj)
                
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                print(f"⚠️ Error processing object {obj.get('NORAD_CAT_ID', 'UNKNOWN')}: {str(e)}")
                continue
        
        print(f"✅ Successfully transformed {len(transformed_objects)} objects")
        return transformed_objects
    
    def _estimate_altitude_from_period(self, period: float) -> float:
        """Estimate altitude from orbital period using Kepler's third law."""
        # Kepler's third law: T^2 = (4π^2/μ) * a^3
        # Where a is semi-major axis
        period_seconds = period * 60
        mu = 398600.4418  # km³/s²
        
        # Calculate semi-major axis
        a = ((mu * (period_seconds / (2 * math.pi))**2)**(1/3))
        
        # Convert to altitude above Earth's surface (average)
        earth_radius = 6371  # km
        altitude = a - earth_radius
        
        # Ensure reasonable bounds
        return max(150, min(altitude, 50000))
    
    def _calculate_simplified_position(self, inclination: float, period: float, 
                                      current_time: float) -> tuple:
        """Calculate simplified latitude/longitude for visualization."""
        # Simplified orbital position calculation
        # Not accurate, but good enough for visualization
        
        # Mean anomaly progression
        period_seconds = period * 60
        mean_anomaly = (current_time % period_seconds) / period_seconds * 2 * math.pi
        
        # Earth rotation
        earth_rotation = (current_time % 86400) / 86400 * 360
        
        # Calculate approximate position
        lat = math.sin(mean_anomaly) * inclination
        lon = (mean_anomaly * 180 / math.pi + earth_rotation) % 360
        
        # Normalize longitude to -180 to 180
        if lon > 180:
            lon -= 360
        
        return lat, lon
    
    def _calculate_3d_coordinates(self, lat: float, lon: float, radius: float) -> tuple:
        """Convert spherical coordinates to 3D Cartesian coordinates."""
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        x = radius * math.cos(lat_rad) * math.cos(lon_rad)
        y = radius * math.cos(lat_rad) * math.sin(lon_rad)
        z = radius * math.sin(lat_rad)
        
        return x, y, z
    
    def _classify_object(self, name: str, norad_id: str) -> str:
        """Classify object type based on name and characteristics."""
        name_upper = name.upper()
        
        if any(keyword in name_upper for keyword in ['DEBRIS', 'DEB', 'FRAGMENT', 'FRAG']):
            return 'DEBRIS'
        elif any(keyword in name_upper for keyword in ['ROCKET', 'R/B', 'STAGE']):
            return 'ROCKET BODY'
        elif any(keyword in name_upper for keyword in ['STARLINK', 'ONEWEB', 'IRIDIUM']):
            return 'CONSTELLATION'
        elif any(keyword in name_upper for keyword in ['ISS', 'STATION', 'TIANGONG']):
            return 'SPACE STATION'
        else:
            return 'PAYLOAD'
    
    def _estimate_size(self, object_type: str, name: str) -> float:
        """Estimate object size based on type and name."""
        # More realistic size estimates
        size_ranges = {
            'DEBRIS': (0.01, 2.0),        # Small debris
            'ROCKET BODY': (5.0, 20.0),    # Large rocket bodies
            'SPACE STATION': (50.0, 100.0), # Very large
            'CONSTELLATION': (2.0, 5.0),   # Small satellites
            'PAYLOAD': (1.0, 15.0)         # Various payload sizes
        }
        
        min_size, max_size = size_ranges.get(object_type, (1.0, 5.0))
        return np.random.uniform(min_size, max_size)
    
    def _calculate_risk_score(self, altitude: float, size: float, object_type: str) -> float:
        """Calculate risk score based on altitude, size, and type."""
        # Normalize altitude risk (lower altitude = higher risk)
        if altitude < 500:  # LEO - high traffic
            altitude_risk = 0.9
        elif altitude < 1000:
            altitude_risk = 0.7
        elif altitude < 2000:
            altitude_risk = 0.5
        elif altitude < 35786:  # Below GEO
            altitude_risk = 0.3
        else:  # GEO and above
            altitude_risk = 0.2
        
        # Size risk (larger = higher risk)
        size_risk = min(size / 20.0, 1.0)
        
        # Type risk weights
        type_risk_weights = {
            'DEBRIS': 0.8,        # Hard to track, unpredictable
            'ROCKET BODY': 0.7,   # Large, often in decaying orbits
            'PAYLOAD': 0.6,       # Operational satellites
            'SPACE STATION': 0.9, # Critical infrastructure
            'CONSTELLATION': 0.5  # Densely populated orbits
        }
        
        type_risk = type_risk_weights.get(object_type, 0.5)
        
        # Weighted average
        weights = {'altitude': 0.4, 'size': 0.3, 'type': 0.3}
        risk_score = (
            altitude_risk * weights['altitude'] +
            size_risk * weights['size'] +
            type_risk * weights['type']
        )
        
        return min(max(risk_score, 0.0), 1.0)

def fetch_celestrak_data(
    include_debris: bool = True, 
    include_starlink: bool = True,
    max_objects: int = 500
) -> List[Dict[str, Any]]:
    """
    Main function to fetch comprehensive satellite and debris data from CelesTrak.
    
    Args:
        include_debris: Whether to include specific debris data
        include_starlink: Whether to include Starlink constellation
        max_objects: Maximum number of objects to return (for performance)
        
    Returns:
        List of all satellite and debris objects
    """
    client = CelesTrakClient()
    
    try:
        print("🌍 Fetching CelesTrak data...")
        
        # Try JSON first, fall back to TLE
        try:
            print("🔄 Attempting to fetch JSON format...")
            active_data = client.fetch_active_satellites(format_type="json")
        except Exception as json_error:
            print(f"⚠️ JSON fetch failed: {json_error}")
            print("🔄 Falling back to TLE format...")
            active_data = client.fetch_active_satellites(format_type="tle")
        
        # Limit total objects
        if len(active_data) > max_objects:
            print(f"⚠️ Limiting to {max_objects} objects (from {len(active_data)})")
            active_data = active_data[:max_objects]
        
        # Transform to dashboard format
        transformed_data = client.transform_to_dashboard_format(active_data)
        
        # Print summary
        print(f"\n📊 CELESTRAK DATA SUMMARY:")
        print(f"   📡 Total Objects: {len(transformed_data)}")
        
        # Count by type
        type_counts = {}
        for obj in transformed_data:
            obj_type = obj.get('object_type', 'UNKNOWN')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        for obj_type, count in sorted(type_counts.items()):
            print(f"   🔹 {obj_type}: {count}")
        
        # Show altitude distribution
        altitudes = [obj['altitude'] for obj in transformed_data]
        if altitudes:
            print(f"   📈 Altitude range: {min(altitudes):.0f} - {max(altitudes):.0f} km")
        
        return transformed_data
        
    except Exception as e:
        print(f"❌ Error fetching CelesTrak data: {str(e)}")
        # Return sample data for testing if real data fails
        return get_sample_data()

def get_sample_data() -> List[Dict[str, Any]]:
    """Return sample data for testing when real data is unavailable."""
    print("⚠️ Using sample data for demonstration")
    
    sample_objects = [
        {
            'id': 'CT-25544',
            'norad_id': '25544',
            'object_name': 'ISS (ZARYA)',
            'object_type': 'SPACE STATION',
            'altitude': 420.0,
            'latitude': 28.5,
            'longitude': -80.2,
            'x': -4500.0,
            'y': 3200.0,
            'z': 4200.0,
            'size': 80.0,
            'velocity': 7.66,
            'inclination': 51.64,
            'eccentricity': 0.001,
            'period': 92.9,
            'mean_motion': 15.5,
            'risk_score': 0.8,
            'last_updated': datetime.now().isoformat(),
            'data_source': 'CelesTrak (Sample)',
            'confidence': 0.9
        },
        {
            'id': 'CT-49260',
            'norad_id': '49260',
            'object_name': 'STARLINK-3000',
            'object_type': 'CONSTELLATION',
            'altitude': 550.0,
            'latitude': -15.3,
            'longitude': 120.5,
            'x': 5100.0,
            'y': -2800.0,
            'z': 3900.0,
            'size': 3.5,
            'velocity': 7.58,
            'inclination': 53.0,
            'eccentricity': 0.001,
            'period': 95.6,
            'mean_motion': 15.1,
            'risk_score': 0.4,
            'last_updated': datetime.now().isoformat(),
            'data_source': 'CelesTrak (Sample)',
            'confidence': 0.9
        }
    ]
    
    return sample_objects

if __name__ == "__main__":
    # Test the CelesTrak client
    try:
        print("🧪 Testing CelesTrak client...")
        
        # Test with limited number of objects
        data = fetch_celestrak_data(max_objects=100)
        
        print(f"\n✅ Successfully processed {len(data)} objects")
        
        if data:
            print(f"\n📊 First object:")
            for key, value in list(data[0].items())[:10]:  # Show first 10 fields
                print(f"   {key}: {value}")
            
            print(f"\n📊 Object type distribution:")
            types = {}
            for obj in data:
                obj_type = obj.get('object_type', 'UNKNOWN')
                types[obj_type] = types.get(obj_type, 0) + 1
            
            for obj_type, count in types.items():
                print(f"   {obj_type}: {count}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()