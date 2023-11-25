function toggleSwitch() {
      var checkbox = document.querySelector('input[type="checkbox"]');
      const numeroContainer = document.getElementById("numeroContainer");
      const textosContainer = document.getElementById("posiciones");

      if (checkbox.checked) {
        // Code to execute when the switch is toggled ON
        console.log("Switch is ON");
        numeroContainer.style.display = "block";
        textosContainer.style.display = "none";
      } else {
        // Code to execute when the switch is toggled OFF
        console.log("Switch is OFF");
        numeroContainer.style.display = "none";
        textosContainer.style.display = "block";
      }
}

function openNewWindow2() {         
  var childWindow = window.open("/open-window2", "NNWindow", "width=500,height=500");
}

function descargarVideo() {
  const link = document.createElement('a');
  link.href = 'guia.mp4';
  link.download = 'video.mp4'; // Nombre del archivo de video que se descargará
}

