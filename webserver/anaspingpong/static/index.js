let map;

var data_xml = '/data'
var newLat = 52.4907
var newLng = 13.4726
var zoom = 18;
var home = true

//let image;
function setCoordinates() {

     if (document.getElementById('center_lat').value != "") {
        newLat = parseFloat(document.getElementById('center_lat').value)
        newLng = parseFloat(document.getElementById('center_lon').value)
        zoom = parseInt(document.getElementById('zoom').value);
        home = false;

    }

}


function placeMarker(location) {
    var image = new google.maps.MarkerImage(
        'static/images/marker.png',
        null, // size
        null, // origin
        new google.maps.Point( 8, 8 ), // anchor (move to center of marker)
        new google.maps.Size( 32, 42 ) // scaled size (required for Retina display icon)
    );

    var pulseMarker = new google.maps.Marker({
        flat: true,
        optimized: false,
        visible: true,
        position: location,
        map: map,
        icon: image,
        title: "ping pong table"
    });
  }

  
function initMap() {
    setCoordinates()

    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: newLat, lng: newLng },
        zoom: zoom,
        tilt: 0,
    });
    map.setMapTypeId(google.maps.MapTypeId.HYBRID);

    if (home == true) {
        goHome();
    }

   getPositions()

 // var directionsService = new google.maps.DirectionsService();
 // var directionsRenderer = new google.maps.DirectionsRenderer();
 // directionsRenderer.setMap(map);

}


function goHome() {
    if(navigator.geolocation) {
        browserSupportFlag = true;
        navigator.geolocation.getCurrentPosition(function(position) {
           currentLocation = new google.maps.LatLng(
               position.coords.latitude,
               position.coords.longitude
            );
        map.setZoom(18)
        map.setCenter(currentLocation);
        });
    }
}

function fillLocation() {
    document.getElementById('location').value = map.getCenter()
    document.getElementById('zoom').value = map.getZoom()
    document.forms[0].submit()
}

function getPositions() {
    markerArray = new Array();
    var jqxhr = $.get(data_xml, function(data) {

          $(data).find("marker").each(function() {
            var marker = jQuery(this);
            var id = marker.attr("id");
            var point = new google.maps.LatLng(parseFloat(marker.attr("lat")),
                                        parseFloat(marker.attr("lng")));


            markerArray.push(placeMarker(point));

          });
    });
}








