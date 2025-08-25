<?php


$id=$_GET['id'];

include 'dbconnect.php';
$sql="UPDATE `login` SET status = '1' WHERE email = '$id'";
if(mysql_query($sql,$conn))
	echo "<script>alert(' Approved!');location.href='a_ureq.php';</script>";
     else
     	{"<script>alert('error!');location.href='a_ureq.php';</script>";
		}

?>