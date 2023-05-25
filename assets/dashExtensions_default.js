window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, latlng, context) {
            const p = feature.properties;
            if (p.type === 'circlemarker') {
                return L.circleMarker(latlng, radius = p._radius)
            }
            if (p.type === 'circle') {
                return L.circle(latlng, radius = p._mRadius)
            }
            return L.marker(latlng);
        }
    }
});