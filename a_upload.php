<?php
  include('admin_nav.php');
  include_once('dbconnect.php');

?>


<form method='POST' enctype='multipart/form-data'>
<center><br><br><br><br><br><br><br><br>
<b>Upload CSV FILE: </b><br><br><input type='file' name='csv_info' /><br><br>
<input type='submit' name='submit' value='Upload' style="height: 50px; width: 150px; left: 250; top: 250;"/>
</center>
</form>
<?php

if(isset($_POST['submit'])){
if($_FILES['csv_info']['name']){
  $arrFileName = explode('.',$_FILES['csv_info']['name']);
   if($arrFileName[1] == 'csv'){
     $handle = fopen($_FILES['csv_info']['tmp_name'], "r");
      while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
         $id=$data[0];
         $result = $data[2];
     
         $sql="update feedback set label='$result' where id='$id'";
 mysql_query($sql);
                                   }
                        fclose($handle);
}
}
}
?>