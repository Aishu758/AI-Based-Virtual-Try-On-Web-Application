<?php
 include('dbconnect.php');

$id=$_GET['id'];

 
  $q="update purchase set status='1' where id='$id'";
  // echo $q;
  if(mysql_query($q,$conn))
  {
    
          echo "<script>alert('Updated!');
              location.href='sorders.php';
          </script>";
        }


?>