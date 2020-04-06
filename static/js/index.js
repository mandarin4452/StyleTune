function readUpload(inputObject) {
    console.log("JS");
    if (window.File && window.FileReader) {
        if (inputObject.files && inputObject.files[0]){
            var reader = new FileReader();
            reader.onload = function(e){
                $('#imagePreview').attr('src',e.target.result);
            }
            reader.readAsDataURL(intput.files[0]);
        }
    }
}
console.log("asdfasd");
$('#uploadImage').change(function(){
    readUpload(this);
})