import streamlit as st

st.set_page_config(page_title="map", page_icon="static/assets/favicon.ico", layout="wide")

st.header("map")
data = {
    'latitude': [37.7749, 34.0522, 40.7128],
    'longitude': [-122.4194, -118.2437, -74.0060],
    'name': ['San Francisco', 'Los Angeles', 'New York'],
}
st.map(data, zoom=4, use_container_width=True)
