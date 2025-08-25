<?php
include('dbconnect.php');
$id=$_GET['id'];

$sql="delete From product where id=$id";

                                 //echo"$sql";
                                 $result=mysql_query($sql,$conn);
                                if($result)
                                {
                                    echo "<script>alert('PRODUCT DELETED!');location.href='smyproduct.php';</script>";
                                }
                                else
                                {
                                    echo "<script>alert('Can't Delete Product!');location.href='smyproduct.php';</script>";
                                }
                                ?>
?>

