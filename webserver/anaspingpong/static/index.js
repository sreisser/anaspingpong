var map;

var data_xml = '/data'
var pin_file = 'static/images/marker.png'
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

function showAddress(address)
{
	  $.getJSON('https://nominatim.openstreetmap.org/search?format=json&limit=1&q=' + address, function(data) {

		  var items = [];
		  var zaehler = 0;
		  var lat = 0;
		  var lng = 0;

			$.each(data, function(key, val) {



			  if (zaehler == 0)
			  {
				lat = val.lat;
			  	lng = val.lon;
			  	//console.log(lat+","+lng);

			  	newLocation = new Microsoft.Maps.Location(lat, lng);
			  	map.setView({
                    center: newLocation,
                    zoom: zoom
                 });

			  }
			  zaehler++;
		  });
	});

}



function GetMap() {
        setCoordinates()

        map = new Microsoft.Maps.Map('#map', {
            credentials: 'AqIKf2KZC7FjrPGfs2eWrDaw2hEj8H3ul8VNA8M6omMTsQ0k0Jha38F6PVtCnBE5',
            center: new Microsoft.Maps.Location(newLat, newLng),
            mapTypeId: Microsoft.Maps.MapTypeId.aerial,
            zoom: zoom,
        });

      //  var center = map.getCenter();

       getPositions(map)
        //Add your post map load code here.
}
  


function goHome() {
    if(navigator.geolocation) {
        browserSupportFlag = true;
        navigator.geolocation.getCurrentPosition(function(position) {
           currentLocation = new Microsoft.Maps.Location(
               position.coords.latitude,
               position.coords.longitude
            );

            map.setView({
                center: currentLocation,
                zoom: 18
            });
      });

    }
}


function fillLocation() {

    lat = map.getCenter().latitude
    lon = map.getCenter().longitude
    zoom = map.getZoom()

    document.getElementById('center_lat').value = lat
    document.getElementById('center_lon').value = lon
    document.getElementById('zoom').value = zoom
    document.getElementById('predicting').style.display = 'block'
    document.forms[0].submit()
}

function getPositions(map) {
   // markerArray = new Array();

    var jqxhr = $.get(data_xml, function(data) {

          $(data).find("marker").each(function() {
            var marker = jQuery(this);
            var id = marker.attr("id");
            loc = new Microsoft.Maps.Location(marker.attr('lat'),
            marker.attr('lng'))

            var point = new Microsoft.Maps.Pushpin(loc, {
                icon: pin_file,
                anchor: new Microsoft.Maps.Point(8, 8),
            });

            map.entities.push(point);

          });
    });
}








