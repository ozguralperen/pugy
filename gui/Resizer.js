/**
 *  Resizer Algorithm of Video Container Block 
 *  
 *  Written by kafein team.
 * 
 * Case 1 : %2 != 0
 *  P1 P2
 *    P3
 * 
 * Case 2 : %2 == 0
 *  P1 P2
 *  P3 P4 
 *  
 * 
 *  BAŞTA EKRANDA 2 KAMERA OLACAK. SONRASINI HALLEDİYOR BU ARKADAŞ....
 */


// Motion Constants 
// (T : -y All Cameras in Top Axis)
// (L : +x All Cameras in Left Axis)
const BETA_T = 100;
const BETA_L = 100;

function AddNewCamera() {
   var nLeft , nTop;
    var NewDiv = document.createElement('video');
    document.body.appendChild(NewDiv);
    NewDiv.className = "camerabox";
    navigator.mediaDevices.getUserMedia(constraintObj)
        .then(function(mediaStreamObj) {
            //connect the media stream to the first video element

            if ("srcObject" in NewDiv) {
                NewDiv.srcObject = mediaStreamObj;
            } else {
                //old version
                NewDiv.src = window.URL.createObjectURL(mediaStreamObj);
            }

            NewDiv.onloadedmetadata = function(ev) {
                //show in the video element what is being captured by the webcam
                NewDiv.play();
            };
        })
   if (($('.camerabox').length + 1) % 2 != 0) {
      $('.camerabox').each(function () {
         var EachCenter = $(this).position();
         EachCenter.top -= BETA_T;
      });
      var LastRight = $('.camerabox').before($('.camerabox').last()).position();
      var LastLeft = $('.camerabox').last().position();

      nLeft = (LastRight.left - LastLeft.left) / 2;
      nTop = LastLeft.top + 2 * BETA_T;
      console.log("New Camera : " + nLeft.toString() +  ".."+ nTop.toString());
   }
   else {
      var LastOneCenter = $('.camerabox').last().position();
      LastOneCenter.left -= BETA_L;
      nTop = LastOneCenter.top;
      nLeft = LastOneCenter.left + 2 * BETA_L;
      
   }
   NewDiv.id = "Camera" + $('.camerabox').length.toString()
   $(NewDiv.id).css({ 'top': nTop, 'margin-left': nLeft });
}

function AddObj() {
   
   console.log("ADD OBJ Calisiyor.");

   
   AddNewCamera();
}
