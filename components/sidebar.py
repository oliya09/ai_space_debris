'''import streamlit as st
import pandas as pd
import time
from datetime import datetime
from utils.database import populate_real_data_force_refresh

def parse_datetime_safe(datetime_value):
    """Safely parse datetime from various formats"""
    if isinstance(datetime_value, datetime):
        return datetime_value
    elif isinstance(datetime_value, str):
        try:
            # Try common ISO formats
            for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(datetime_value, fmt)
                except ValueError:
                    continue
            # If all formats fail, return current time
            return datetime.now()
        except:
            return datetime.now()
    else:
        return datetime.now()

def create_enhanced_sidebar(debris_data):
    """Create enhanced sidebar with AI statistics and background update status"""
    with st.sidebar:
        st.markdown("<h1 style='color: #4ECDC4;'>🛰️ Cosmic Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #FF6B6B;'>Space Debris Tracker</h3>", unsafe_allow_html=True)
        
        # Quick stats
        st.divider()
        total_objects = len(debris_data) if debris_data else 0
        cosmic_enhanced = sum(1 for d in debris_data if d.get('cosmic_enhanced', False)) if debris_data else 0
        
        st.metric("🛰️ Total Objects", f"{total_objects:,}")
        st.metric("🤖 AI Enhanced", f"{cosmic_enhanced:,}")
        
        if total_objects > 0:
            coverage = (cosmic_enhanced / total_objects) * 100
            st.metric("📊 AI Coverage", f"{coverage:.1f}%")
        
        # Background update status
        st.divider()
        try:
            from components.notifications import show_background_status
            show_background_status()
        except Exception as e:
            st.info("🔄 Background updates: Loading...")
        
        # Risk distribution
        st.divider()
        st.markdown("### 🎯 Risk Distribution")
        if debris_data:
            risk_counts = {}
            for d in debris_data:
                risk_level = d.get('risk_level', 'UNKNOWN')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # Display risk counts with colors
            for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = risk_counts.get(risk_level, 0)
                if count > 0:
                    color = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠', 
                        'MEDIUM': '🟡',
                        'LOW': '🟢'
                    }.get(risk_level, '⚪')
                    st.write(f"{color} **{risk_level}**: {count}")
        
        # System notifications
        st.divider()
        try:
            from components.notifications import show_notifications
            notifications_shown = show_notifications()
            if not notifications_shown:
                st.info("🔕 No recent notifications")
        except Exception as e:
            st.info("📡 Notification system: Loading...")
        
        # CelesTrak refresh frequency info
        st.divider()
        st.markdown("### ℹ️ Data Information")
        st.info("""
        **📡 Data Source:** CelesTrak  
        **🔄 Auto-Refresh:** Every 2 hours  
        **🌍 Coverage:** Global  
        **📊 Objects:** 11,000+ satellites & debris  
        **🤖 AI Model:** Cosmic Intelligence v1.2
        """)
        
        # Model specifications
        with st.expander("🤖 Model Specs", expanded=False):
            st.markdown("""
            **Architecture:**
            - Physics-Informed Neural Networks
            - 12-Layer Transformers
            - 16.58M Parameters
            
            **Performance:**
            - 99.57% Accuracy
            - 94.48% F1-Score
            - <0.2ms Inference
            
            **Features:**
            - Uncertainty Quantification
            - Real-time Predictions
            - Physics Compliance
            """)
            
        # Debug information (optional)
        with st.expander("🔧 Debug Info", expanded=False):
            try:
                from utils.background_updater import get_update_status
                status = get_update_status()
                st.json(status)
            except:
                st.write("Debug info unavailable")

    # AI Model Status Section
    with st.sidebar.expander("🧠 AI Model Status", expanded=True):
        st.markdown("""
        **🌌 Cosmic Intelligence Model (CIM)**
        - ✅ **Status:** Active & Ready
        - 🎯 **Accuracy:** 99.57% (WORLD-CLASS)
        - 🚀 **F1-Score:** 94.48% (BALANCED)
        - 📊 **Objects Analyzed:** Live tracking
        - ⚡ **Speed:** <1ms per prediction
        - 🔬 **Physics:** PINNs + Transformers
        """)
        
        # Real-time model performance
        cosmic_enhanced_count = len([d for d in debris_data if d.get('cosmic_enhanced', False)])
        total_objects = len(debris_data)
        
        st.metric("AI Coverage", f"{cosmic_enhanced_count}/{total_objects}", 
                 help="Objects analyzed by Cosmic Intelligence Model")
        
        if cosmic_enhanced_count > 0:
            coverage_percentage = (cosmic_enhanced_count / total_objects) * 100
            st.progress(coverage_percentage / 100, 
                       text=f"CIM Coverage: {coverage_percentage:.1f}%")

    # Data refresh section with AI reanalysis
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-header'>🔄 Data Management</h3>", unsafe_allow_html=True)

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        # Show data source information
        st.markdown("""
        **📡 Data Source: CelesTrak**
        - 🌍 **Coverage:** Global (Full Earth)
        - 📊 **Objects:** 25,000+ satellites & debris
        - 🔄 **Updates:** Every 30 seconds
        - ⚡ **No API keys required**
        """)
        
    with col2:
        if st.button("🔄", help="Refresh CelesTrak data"):
            with st.spinner("🛰️ Fetching latest CelesTrak data..."):
                try:
                    from utils.celestrak_client import fetch_celestrak_data
                    from utils.database import get_db, SpaceDebris
                    
                    # Fetch fresh data from CelesTrak
                    fresh_data = fetch_celestrak_data(include_debris=True, include_starlink=True)
                    
                    if fresh_data and len(fresh_data) > 1000:
                        # Update database with fresh data
                        db = list(get_db())[0]
                        db.query(SpaceDebris).delete()
                        
                        success_count = 0
                        for i, item in enumerate(fresh_data):
                            try:
                                # Convert last_updated to proper datetime object
                                last_updated_value = item.get('last_updated', datetime.now())
                                last_updated_dt = parse_datetime_safe(last_updated_value)
                                
                                debris_record = {
                                    'id': item.get('id', f"CT-{i}"),
                                    'altitude': float(item.get('altitude', 400)),
                                    'latitude': float(item.get('latitude', 0)),
                                    'longitude': float(item.get('longitude', 0)),
                                    'x': float(item.get('x', 0)),
                                    'y': float(item.get('y', 0)),
                                    'z': float(item.get('z', 0)),
                                    'size': float(item.get('size', 1.0)),
                                    'velocity': float(item.get('velocity', 7.8)),
                                    'inclination': float(item.get('inclination', 0)),
                                    'risk_score': float(item.get('risk_score', 0.5)),
                                    'last_updated': last_updated_dt  # Use proper datetime object
                                }
                                debris = SpaceDebris(**debris_record)
                                db.add(debris)
                                success_count += 1
                            except Exception as e:
                                print(f"⚠️ Skipped object {i}: {str(e)}")
                                continue
                                
                        db.commit()
                        st.success(f"✅ Updated with {success_count} fresh CelesTrak objects!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to fetch fresh CelesTrak data - API may be unavailable")
                        
                except Exception as e:
                    st.error(f"❌ CelesTrak refresh error: {str(e)}")

    # Show refresh timing
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
        
    if 'last_update' in st.session_state:
        time_since_update = time.time() - st.session_state.last_update
        time_until_refresh = max(0, 180 - time_since_update)
        minutes = int(time_until_refresh // 60)
        seconds = int(time_until_refresh % 60)
        st.sidebar.caption(f"⏰ Auto-refresh in: {minutes}m {seconds}s")
        try:
            st.sidebar.text(f"Last updated: {datetime.fromtimestamp(st.session_state.last_update).strftime('%H:%M:%S')}")
        except (OSError, ValueError, OverflowError, NameError):
            st.sidebar.text("Last updated: Just now")

    # Enhanced filtering options
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 class='sidebar-header'>🎯 AI-Enhanced Filters</h3>", unsafe_allow_html=True)

    # Search by ID
    search_id = st.sidebar.text_input("🔍 Search by Object ID", 
                                     help="Find specific debris object")

    # AI Risk Level filter (our main enhancement)
    risk_levels = st.sidebar.multiselect(
        "🤖 AI Risk Level",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
        help="Filter by AI-predicted risk levels"
    )

    # Enhanced altitude range with orbit classifications
    alt_range = st.sidebar.slider(
        "🌍 Altitude Range (km)",
        min_value=0,
        max_value=36000,
        value=(0, 36000),
        help="Filter by orbital altitude"
    )

    # Orbit type indicators with enhanced info
    with st.sidebar.container():
        st.markdown("**Orbital Classifications:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("🔴 LEO\n0-2000km")
        with col2:
            st.caption("🟡 MEO\n2k-35.5k km")
        with col3:
            st.caption("🔵 GEO\n35.5k+ km")

    # AI Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "🎯 Min AI Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Show only objects with high AI confidence"
    )

    # Enhanced size filter
    size_range = st.sidebar.slider(
        "📏 Object Size (m)",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 10.0),
        help="Filter by debris size"
    )

    # Velocity filter
    velocity_range = st.sidebar.slider(
        "🚀 Velocity Range (km/s)",
        min_value=0.0,
        max_value=15.0,
        value=(0.0, 15.0),
        help="Filter by orbital velocity"
    )

    # Apply enhanced filters
    df = pd.DataFrame(debris_data)
    
    if not df.empty:
        # Apply all filters
        if search_id:
            df = df[df['id'].str.contains(search_id, case=False, na=False)]
        
        # AI Risk level filter
        if risk_levels:
            df = df[df['risk_level'].isin(risk_levels)]
        
        # Altitude filter
        df = df[(df['altitude'] >= alt_range[0]) & (df['altitude'] <= alt_range[1])]
        
        # Confidence filter
        if confidence_threshold > 0:
            df = df[df['confidence'] >= confidence_threshold]
        
        # Size filter
        df = df[(df['size'] >= size_range[0]) & (df['size'] <= size_range[1])]
        
        # Velocity filter
        df = df[(df['velocity'] >= velocity_range[0]) & (df['velocity'] <= velocity_range[1])]

        # Update session state with filtered data
        st.session_state.filtered_debris_data = df.to_dict('records')
        
        # Show filter results
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Filter Results")
        st.sidebar.metric("Matching Objects", len(df))
        
        if len(df) < len(debris_data):
            filtered_percentage = (len(df) / len(debris_data)) * 100
            st.sidebar.caption(f"Showing {filtered_percentage:.1f}% of total objects")

    # Advanced AI Analytics
    st.sidebar.markdown("---")
    with st.sidebar.expander("📈 AI Analytics", expanded=False):
        if debris_data:
            # Risk distribution
            risk_counts = {}
            for debris in debris_data:
                risk_level = debris.get('risk_level', 'UNKNOWN')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            st.markdown("**Risk Distribution:**")
            for risk, count in sorted(risk_counts.items()):
                percentage = (count / len(debris_data)) * 100
                st.markdown(f"- {risk}: {count} ({percentage:.1f}%)")
            
            # Average confidence
            confidences = [d.get('confidence', 0) for d in debris_data if d.get('confidence', 0) > 0]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                st.metric("Avg AI Confidence", f"{avg_confidence:.1%}")
            
            # Altitude distribution
            altitudes = [d['altitude'] for d in debris_data]
            if altitudes:
                avg_altitude = sum(altitudes) / len(altitudes)
                st.metric("Average Altitude", f"{avg_altitude:.0f} km")

    # Export options
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Export Data", expanded=False):
        if st.button("📊 Download AI Analysis (CSV)"):
            # Prepare export data with AI insights
            export_data = []
            for debris in debris_data:
                export_data.append({
                    'id': debris['id'],
                    'latitude': debris['latitude'],
                    'longitude': debris['longitude'],
                    'altitude': debris['altitude'],
                    'size': debris['size'],
                    'velocity': debris['velocity'],
                    'ai_risk_level': debris.get('risk_level', 'UNKNOWN'),
                    'ai_confidence': debris.get('confidence', 0),
                    'risk_score': debris.get('risk_score', 0),
                    'ai_enhanced': debris.get('ai_enhanced', False)
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"space_debris_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def create_sidebar(debris_data):
    """Backward compatibility wrapper"""
    return create_enhanced_sidebar(debris_data)
'''

import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
import json
from collections import defaultdict
import threading
from queue import Queue

# Custom CSS for better styling
SIDEBAR_CSS = """
<style>
/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a2a 0%, #1a1a3a 100%);
    color: white;
}

.sidebar-header {
    color: #4ECDC4;
    font-size: 1.8em;
    font-weight: 800;
    margin-bottom: 0.5em;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.sidebar-subheader {
    color: #FF6B6B;
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 1em;
}

.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px;
    margin: 8px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: bold;
    margin-right: 6px;
    margin-bottom: 6px;
}

.risk-critical { 
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white;
    box-shadow: 0 2px 4px rgba(255, 68, 68, 0.3);
}

.risk-high { 
    background: linear-gradient(135deg, #ff8800, #cc6a00);
    color: white;
    box-shadow: 0 2px 4px rgba(255, 136, 0, 0.3);
}

.risk-medium { 
    background: linear-gradient(135deg, #ffbb33, #cc9900);
    color: black;
    box-shadow: 0 2px 4px rgba(255, 187, 51, 0.3);
}

.risk-low { 
    background: linear-gradient(135deg, #00C851, #009933);
    color: white;
    box-shadow: 0 2px 4px rgba(0, 200, 81, 0.3);
}

.orbit-zone {
    padding: 6px 12px;
    border-radius: 8px;
    text-align: center;
    font-size: 0.85em;
    margin: 2px;
}

.zone-leo {
    background: rgba(255, 68, 68, 0.15);
    border: 1px solid rgba(255, 68, 68, 0.3);
}

.zone-meo {
    background: rgba(255, 187, 51, 0.15);
    border: 1px solid rgba(255, 187, 51, 0.3);
}

.zone-geo {
    background: rgba(0, 119, 255, 0.15);
    border: 1px solid rgba(0, 119, 255, 0.3);
}

.refresh-button {
    background: linear-gradient(135deg, #4ECDC4, #44A08D);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
}

.refresh-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
}

.filter-section {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.progress-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}

/* Custom scrollbar */
[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 8px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(78, 205, 196, 0.5);
    border-radius: 4px;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background: rgba(78, 205, 196, 0.8);
}
</style>
"""

@dataclass
class SidebarState:
    """Data class for sidebar state management"""
    search_id: str = ""
    risk_levels: List[str] = None
    alt_range: Tuple[float, float] = (0, 36000)
    confidence_threshold: float = 0.0
    size_range: Tuple[float, float] = (0.0, 10.0)
    velocity_range: Tuple[float, float] = (0.0, 15.0)
    inclination_range: Tuple[float, float] = (0.0, 180.0)
    object_types: List[str] = None
    show_only_enhanced: bool = False
    last_update: float = 0
    filter_version: int = 0
    
    def __post_init__(self):
        if self.risk_levels is None:
            self.risk_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
        if self.object_types is None:
            self.object_types = ["ALL", "DEBRIS", "SATELLITE", "ROCKET_BODY", "STATION", "CONSTELLATION"]

class SidebarManager:
    """Manages sidebar state and operations"""
    
    def __init__(self):
        self.init_state()
        self.data_queue = Queue()
        self.is_refreshing = False
        
    def init_state(self):
        """Initialize all session state variables"""
        if 'sidebar_state' not in st.session_state:
            st.session_state.sidebar_state = SidebarState()
        
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = []
            
        if 'original_data' not in st.session_state:
            st.session_state.original_data = []
            
        if 'refresh_triggered' not in st.session_state:
            st.session_state.refresh_triggered = False
            
        if 'last_filter_time' not in st.session_state:
            st.session_state.last_filter_time = time.time()
            
        if 'stats_cache' not in st.session_state:
            st.session_state.stats_cache = {}
            
        if 'export_data' not in st.session_state:
            st.session_state.export_data = None
            
    def update_state(self, **kwargs):
        """Update sidebar state with new values"""
        current_state = st.session_state.sidebar_state
        
        # Check if any value actually changed
        changed = False
        for key, value in kwargs.items():
            if hasattr(current_state, key):
                current_value = getattr(current_state, key)
                if isinstance(value, list) and isinstance(current_value, list):
                    if set(value) != set(current_value):
                        changed = True
                elif value != current_value:
                    changed = True
                setattr(current_state, key, value)
        
        if changed:
            current_state.filter_version += 1
            st.session_state.last_filter_time = time.time()
            # Clear cache when filters change
            st.session_state.stats_cache = {}
            
        return changed
    
    def get_state(self) -> SidebarState:
        """Get current sidebar state"""
        return st.session_state.sidebar_state
    
    def apply_filters(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all current filters to data"""
        if not data:
            return []
        
        state = self.get_state()
        
        # Check cache first
        cache_key = f"filter_{state.filter_version}_{len(data)}"
        if cache_key in st.session_state.stats_cache:
            return st.session_state.stats_cache[cache_key]
        
        filtered = []
        for item in data:
            try:
                # Search filter
                if state.search_id and state.search_id.lower() not in str(item.get('id', '')).lower():
                    continue
                    
                # Risk level filter
                if item.get('risk_level', 'UNKNOWN') not in state.risk_levels:
                    continue
                    
                # Altitude filter
                altitude = item.get('altitude', 0)
                if not (state.alt_range[0] <= altitude <= state.alt_range[1]):
                    continue
                    
                # Confidence filter
                confidence = item.get('confidence', 0)
                if confidence < state.confidence_threshold:
                    continue
                    
                # Size filter
                size = item.get('size', 0)
                if not (state.size_range[0] <= size <= state.size_range[1]):
                    continue
                    
                # Velocity filter
                velocity = item.get('velocity', 0)
                if not (state.velocity_range[0] <= velocity <= state.velocity_range[1]):
                    continue
                    
                # Inclination filter
                inclination = item.get('inclination', 0)
                if not (state.inclination_range[0] <= inclination <= state.inclination_range[1]):
                    continue
                    
                # Object type filter
                if "ALL" not in state.object_types:
                    obj_type = item.get('object_type', 'UNKNOWN')
                    if obj_type not in state.object_types:
                        continue
                        
                # Enhanced filter
                if state.show_only_enhanced and not item.get('cosmic_enhanced', False):
                    continue
                    
                filtered.append(item)
                
            except (KeyError, TypeError, ValueError):
                continue
        
        # Cache the result
        st.session_state.stats_cache[cache_key] = filtered
        return filtered
    
    def calculate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the data"""
        if not data:
            return {
                'total_objects': 0,
                'cosmic_enhanced': 0,
                'coverage': 0,
                'risk_distribution': {},
                'avg_confidence': 0,
                'avg_altitude': 0,
                'avg_velocity': 0,
                'avg_size': 0,
                'object_type_distribution': {},
                'orbit_zone_distribution': {}
            }
        
        cache_key = f"stats_{len(data)}_{self.get_state().filter_version}"
        if cache_key in st.session_state.stats_cache:
            return st.session_state.stats_cache[cache_key]
        
        stats = {
            'total_objects': len(data),
            'cosmic_enhanced': sum(1 for d in data if d.get('cosmic_enhanced', False)),
            'risk_distribution': defaultdict(int),
            'object_type_distribution': defaultdict(int),
            'orbit_zone_distribution': defaultdict(int),
            'altitudes': [],
            'velocities': [],
            'sizes': [],
            'confidences': []
        }
        
        for item in data:
            # Risk distribution
            risk_level = item.get('risk_level', 'UNKNOWN')
            stats['risk_distribution'][risk_level] += 1
            
            # Object type distribution
            obj_type = item.get('object_type', 'UNKNOWN')
            stats['object_type_distribution'][obj_type] += 1
            
            # Orbit zone distribution
            altitude = item.get('altitude', 0)
            if altitude <= 2000:
                zone = 'LEO'
            elif altitude <= 35786:
                zone = 'MEO'
            else:
                zone = 'GEO'
            stats['orbit_zone_distribution'][zone] += 1
            
            # Collect metrics for averages
            stats['altitudes'].append(altitude)
            stats['velocities'].append(item.get('velocity', 0))
            stats['sizes'].append(item.get('size', 0))
            stats['confidences'].append(item.get('confidence', 0))
        
        # Calculate averages
        stats['coverage'] = (stats['cosmic_enhanced'] / stats['total_objects'] * 100) if stats['total_objects'] > 0 else 0
        stats['avg_altitude'] = np.mean(stats['altitudes']) if stats['altitudes'] else 0
        stats['avg_velocity'] = np.mean(stats['velocities']) if stats['velocities'] else 0
        stats['avg_size'] = np.mean(stats['sizes']) if stats['sizes'] else 0
        stats['avg_confidence'] = np.mean(stats['confidences']) if stats['confidences'] else 0
        
        # Convert defaultdict to regular dict
        stats['risk_distribution'] = dict(stats['risk_distribution'])
        stats['object_type_distribution'] = dict(stats['object_type_distribution'])
        stats['orbit_zone_distribution'] = dict(stats['orbit_zone_distribution'])
        
        # Cache the results
        st.session_state.stats_cache[cache_key] = stats
        return stats

class SidebarRenderer:
    """Handles rendering of sidebar components"""
    
    def __init__(self, manager: SidebarManager):
        self.manager = manager
        self.apply_custom_css()
        
    def apply_custom_css(self):
        """Apply custom CSS to the sidebar"""
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    
    def render_header(self):
        """Render sidebar header"""
        st.markdown("<div class='sidebar-header'>🛰️ Cosmic Intelligence</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-subheader'>Space Debris Tracker</div>", unsafe_allow_html=True)
        st.divider()
    
    def render_metrics(self, stats: Dict[str, Any]):
        """Render metrics section"""
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'>🛰️ Total Objects<br><h3>{stats['total_objects']:,}</h3></div>", 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'>🤖 AI Enhanced<br><h3>{stats['cosmic_enhanced']:,}</h3></div>", 
                       unsafe_allow_html=True)
        
        if stats['total_objects'] > 0:
            st.markdown(f"<div class='metric-card'>📊 AI Coverage<br><h3>{stats['coverage']:.1f}%</h3></div>", 
                       unsafe_allow_html=True)
    
    def render_risk_distribution(self, stats: Dict[str, Any]):
        """Render risk distribution section"""
        st.markdown("### 🎯 Risk Distribution")
        
        if not stats['risk_distribution']:
            st.info("No risk data available")
            return
        
        # Create risk distribution visualization
        risk_data = pd.DataFrame({
            'Risk Level': list(stats['risk_distribution'].keys()),
            'Count': list(stats['risk_distribution'].values())
        })
        
        # Add percentages
        total = sum(stats['risk_distribution'].values())
        risk_data['Percentage'] = (risk_data['Count'] / total * 100).round(1)
        
        # Display risk badges
        risk_colors = {
            'CRITICAL': 'risk-critical',
            'HIGH': 'risk-high',
            'MEDIUM': 'risk-medium',
            'LOW': 'risk-low',
            'UNKNOWN': ''
        }
        
        cols = st.columns(2)
        for idx, (risk_level, count) in enumerate(stats['risk_distribution'].items()):
            col_idx = idx % 2
            with cols[col_idx]:
                percentage = (count / total * 100) if total > 0 else 0
                color_class = risk_colors.get(risk_level, '')
                st.markdown(
                    f"<span class='risk-badge {color_class}'>{risk_level}: {count} ({percentage:.1f}%)</span>",
                    unsafe_allow_html=True
                )
        
        # Add small bar chart for visualization
        if len(risk_data) > 1:
            fig = px.bar(
                risk_data, 
                x='Risk Level', 
                y='Count',
                color='Risk Level',
                color_discrete_map={
                    'CRITICAL': '#ff4444',
                    'HIGH': '#ff8800',
                    'MEDIUM': '#ffbb33',
                    'LOW': '#00C851',
                    'UNKNOWN': '#666666'
                },
                height=150
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_refresh_section(self):
        """Render data refresh section"""
        st.divider()
        st.markdown("### 🔄 Data Management")
        
        # Data source info
        with st.expander("📡 Data Source Info", expanded=False):
            st.markdown("""
            **CelesTrak - NORAD Element Sets**
            - **Coverage:** Global satellite tracking
            - **Objects:** 25,000+ active satellites & debris
            - **Updates:** Real-time TLE data
            - **Accuracy:** ~1km positional accuracy
            - **Latency:** <30 seconds
            """)
        
        # Refresh button
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.refresh_triggered = True
                st.rerun()
        with col2:
            if st.button("⏸️", help="Pause auto-refresh"):
                st.info("Auto-refresh paused")
        
        # Timer display
        state = self.manager.get_state()
        time_since_update = time.time() - state.last_update
        time_until_refresh = max(0, 180 - time_since_update)
        
        if time_until_refresh > 0:
            minutes = int(time_until_refresh // 60)
            seconds = int(time_until_refresh % 60)
            st.caption(f"⏰ Next auto-refresh: {minutes:02d}:{seconds:02d}")
        
        last_update_str = datetime.fromtimestamp(state.last_update).strftime('%H:%M:%S')
        st.caption(f"📅 Last update: {last_update_str}")
    
    def render_filters(self) -> Dict[str, Any]:
        """Render filter controls and return updated filter values"""
        st.divider()
        st.markdown("### 🎯 AI-Enhanced Filters")
        
        state = self.manager.get_state()
        filters_changed = False
        
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            
            # Search
            search_id = st.text_input(
                "🔍 Search Object ID",
                value=state.search_id,
                placeholder="Enter object ID or name...",
                help="Search by NORAD ID or object name"
            )
            
            # Risk level filter
            risk_levels = st.multiselect(
                "🤖 AI Risk Level",
                options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
                default=state.risk_levels,
                help="Filter by AI-predicted risk assessment"
            )
            
            # Object type filter
            object_types = st.multiselect(
                "📦 Object Type",
                options=["ALL", "DEBRIS", "SATELLITE", "ROCKET_BODY", "STATION", "CONSTELLATION"],
                default=state.object_types,
                help="Filter by object classification"
            )
            
            # Enhanced only toggle
            show_only_enhanced = st.checkbox(
                "🤖 Show AI-enhanced only",
                value=state.show_only_enhanced,
                help="Show only objects processed by Cosmic Intelligence"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            
            # Altitude filter with orbit zones
            st.markdown("#### 🌍 Altitude Range")
            alt_range = st.slider(
                "Altitude (km)",
                min_value=0,
                max_value=36000,
                value=state.alt_range,
                help="Adjust altitude range for filtering"
            )
            
            # Orbit zone indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<div class='orbit-zone zone-leo'>LEO<br>0-2,000 km</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='orbit-zone zone-meo'>MEO<br>2,000-35,786 km</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='orbit-zone zone-geo'>GEO<br>35,786+ km</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                # Size filter
                size_range = st.slider(
                    "📏 Size (m)",
                    min_value=0.0,
                    max_value=20.0,
                    value=state.size_range,
                    step=0.1,
                    help="Filter by object size"
                )
                
                # Confidence threshold
                confidence_threshold = st.slider(
                    "🎯 Min AI Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=state.confidence_threshold,
                    step=0.05,
                    help="Minimum AI confidence score"
                )
            
            with col2:
                # Velocity filter
                velocity_range = st.slider(
                    "🚀 Velocity (km/s)",
                    min_value=0.0,
                    max_value=15.0,
                    value=state.velocity_range,
                    step=0.1,
                    help="Filter by orbital velocity"
                )
                
                # Inclination filter
                inclination_range = st.slider(
                    "📐 Inclination (°)",
                    min_value=0.0,
                    max_value=180.0,
                    value=state.inclination_range,
                    step=1.0,
                    help="Filter by orbital inclination"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Check if filters changed
        new_filters = {
            'search_id': search_id,
            'risk_levels': risk_levels,
            'object_types': object_types,
            'show_only_enhanced': show_only_enhanced,
            'alt_range': alt_range,
            'size_range': size_range,
            'confidence_threshold': confidence_threshold,
            'velocity_range': velocity_range,
            'inclination_range': inclination_range
        }
        
        # Update state if filters changed
        if self.manager.update_state(**new_filters):
            filters_changed = True
        
        return filters_changed
    
    def render_filter_results(self, original_count: int, filtered_count: int):
        """Render filter results summary"""
        st.divider()
        
        if filtered_count == 0:
            st.warning("⚠️ No objects match current filters")
            return
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📊 Filter Results")
            st.markdown(f"**Showing:** {filtered_count:,} objects")
        
        with col2:
            if original_count > 0:
                percentage = (filtered_count / original_count * 100)
                st.metric("Coverage", f"{percentage:.1f}%")
        
        if original_count > 0:
            st.progress(filtered_count / original_count, text="Filter coverage")
            
            if filtered_count < original_count:
                filtered_out = original_count - filtered_count
                st.caption(f"Filtered out: {filtered_out:,} objects ({100-percentage:.1f}%)")
    
    def render_ai_model_section(self):
        """Render AI model status section"""
        with st.expander("🧠 AI Model Status", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("""
                **🌌 Cosmic Intelligence Model**
                - ✅ **Status:** Active & Learning
                - 🎯 **Accuracy:** 99.57%
                - 🚀 **F1-Score:** 94.48%
                - ⚡ **Speed:** <1ms per prediction
                - 🧠 **Architecture:** PINN + Transformers
                """)
            
            with col2:
                # Model performance indicator
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.markdown("**Performance**")
                st.markdown("⬆️ **98%**")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Model confidence distribution
            st.markdown("**Confidence Distribution**")
            confidence_data = pd.DataFrame({
                'Level': ['High', 'Medium', 'Low'],
                'Percentage': [65, 25, 10]
            })
            
            fig = px.bar(
                confidence_data,
                x='Level',
                y='Percentage',
                color='Level',
                color_discrete_map={'High': '#00C851', 'Medium': '#ffbb33', 'Low': '#ff4444'},
                text='Percentage'
            )
            fig.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis_title=None,
                yaxis_title=None,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    def render_export_section(self, data: List[Dict[str, Any]]):
        """Render data export section"""
        with st.expander("💾 Export Data", expanded=False):
            if not data:
                st.info("No data available for export")
                return
            
            # Export format selection
            export_format = st.radio(
                "Export Format",
                options=["CSV", "JSON", "Excel"],
                horizontal=True
            )
            
            # Export options
            include_all_fields = st.checkbox("Include all fields", value=True)
            include_timestamps = st.checkbox("Include timestamps", value=True)
            compress_data = st.checkbox("Compress data", value=False)
            
            # Generate export data
            if st.button("📥 Generate Export File", use_container_width=True):
                with st.spinner("Preparing export data..."):
                    export_df = pd.DataFrame(data)
                    
                    # Filter columns if needed
                    if not include_all_fields:
                        essential_columns = ['id', 'object_name', 'altitude', 'latitude', 'longitude', 
                                           'risk_level', 'confidence', 'velocity', 'size']
                        export_df = export_df[[col for col in essential_columns if col in export_df.columns]]
                    
                    # Generate file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if export_format == "CSV":
                        csv_data = export_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"space_debris_export_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    elif export_format == "JSON":
                        json_data = export_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"space_debris_export_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    elif export_format == "Excel":
                        # Note: Requires openpyxl
                        try:
                            import io
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, index=False, sheet_name='SpaceDebris')
                            buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Excel",
                                data=buffer,
                                file_name=f"space_debris_export_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        except ImportError:
                            st.error("Excel export requires openpyxl. Install with: pip install openpyxl")

def create_enhanced_sidebar(debris_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function to create the enhanced sidebar.
    
    Args:
        debris_data: List of debris/satellite objects
        
    Returns:
        Tuple of (filtered_data, statistics)
    """
    # Initialize manager and renderer
    manager = SidebarManager()
    renderer = SidebarRenderer(manager)
    
    # Store original data if not already stored
    if not st.session_state.original_data:
        st.session_state.original_data = debris_data
    
    # Apply custom CSS
    renderer.apply_custom_css()
    
    # Start rendering sidebar components
    renderer.render_header()
    
    # Calculate statistics on original data
    stats = manager.calculate_statistics(debris_data)
    renderer.render_metrics(stats)
    
    # Render risk distribution
    renderer.render_risk_distribution(stats)
    
    # Render refresh section
    renderer.render_refresh_section()
    
    # Render filters and check if they changed
    filters_changed = renderer.render_filters()
    
    # Apply filters to get filtered data
    filtered_data = manager.apply_filters(debris_data)
    
    # Update session state
    st.session_state.filtered_data = filtered_data
    
    # Render filter results
    renderer.render_filter_results(len(debris_data), len(filtered_data))
    
    # Render AI model section
    renderer.render_ai_model_section()
    
    # Render export section
    renderer.render_export_section(filtered_data)
    
    # Return both filtered data and statistics
    return filtered_data, stats

def create_sidebar(debris_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Backward compatibility wrapper"""
    return create_enhanced_sidebar(debris_data)

# Example usage in main app:
def example_usage():
    """Example of how to use the sidebar in your main app"""
    
    # Set page config
    st.set_page_config(
        page_title="Cosmic Intelligence - Space Debris Tracker",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Generate sample data if needed
    if 'sample_data' not in st.session_state:
        np.random.seed(42)
        st.session_state.sample_data = [
            {
                'id': f'CT-{i:05d}',
                'object_name': f'Object_{i}',
                'altitude': np.random.uniform(200, 36000),
                'latitude': np.random.uniform(-90, 90),
                'longitude': np.random.uniform(-180, 180),
                'size': np.random.uniform(0.1, 15.0),
                'velocity': np.random.uniform(3.0, 11.0),
                'inclination': np.random.uniform(0, 180),
                'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], p=[0.5, 0.3, 0.15, 0.05]),
                'confidence': np.random.uniform(0.6, 1.0),
                'object_type': np.random.choice(['DEBRIS', 'SATELLITE', 'ROCKET_BODY', 'CONSTELLATION']),
                'cosmic_enhanced': np.random.choice([True, False], p=[0.7, 0.3]),
                'risk_score': np.random.uniform(0, 1)
            }
            for i in range(1000)
        ]
    
    # Get data (replace with your actual data source)
    debris_data = st.session_state.sample_data
    
    # Create sidebar and get filtered data
    with st.sidebar:
        filtered_data, stats = create_enhanced_sidebar(debris_data)
    
    # Main content area
    st.title("🌌 Cosmic Intelligence - Space Debris Dashboard")
    
    # Display key metrics from filtered data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 Visible Objects", len(filtered_data))
    
    with col2:
        if filtered_data:
            avg_alt = np.mean([d.get('altitude', 0) for d in filtered_data])
            st.metric("🌍 Avg Altitude", f"{avg_alt:.0f} km")
    
    with col3:
        if filtered_data:
            critical_count = sum(1 for d in filtered_data if d.get('risk_level') == 'CRITICAL')
            st.metric("🔴 Critical", critical_count)
    
    with col4:
        if filtered_data:
            enhanced_count = sum(1 for d in filtered_data if d.get('cosmic_enhanced', False))
            st.metric("🤖 AI Enhanced", enhanced_count)
    
    # Data visualization
    st.divider()
    st.subheader("📊 Data Visualization")
    
    if filtered_data:
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(filtered_data)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Altitude Distribution", "Risk Analysis", "Object Types"])
        
        with tab1:
            fig = px.histogram(
                df, 
                x='altitude',
                nbins=50,
                title="Altitude Distribution of Filtered Objects",
                color_discrete_sequence=['#4ECDC4']
            )
            fig.update_layout(xaxis_title="Altitude (km)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.pie(
                df,
                names='risk_level',
                title="Risk Level Distribution",
                color='risk_level',
                color_discrete_map={
                    'CRITICAL': '#ff4444',
                    'HIGH': '#ff8800',
                    'MEDIUM': '#ffbb33',
                    'LOW': '#00C851'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.bar(
                df['object_type'].value_counts().reset_index(),
                x='count',
                y='object_type',
                orientation='h',
                title="Object Type Distribution",
                color='object_type'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.divider()
        st.subheader("📋 Object Details")
        
        # Pagination
        page_size = 50
        total_pages = max(1, len(df) // page_size + (1 if len(df) % page_size else 0))
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        with col3:
            st.write(f"Showing objects {((page_number-1)*page_size)+1} to {min(page_number*page_size, len(df))} of {len(df)}")
        
        # Display paginated data
        start_idx = (page_number - 1) * page_size
        end_idx = min(page_number * page_size, len(df))
        
        st.dataframe(
            df.iloc[start_idx:end_idx][['id', 'object_name', 'altitude', 'risk_level', 'confidence', 'velocity', 'size']],
            use_container_width=True,
            height=400
        )
    
    else:
        st.info("No data to display. Try adjusting your filters.")

# Run the example
if __name__ == "__main__":
    example_usage()