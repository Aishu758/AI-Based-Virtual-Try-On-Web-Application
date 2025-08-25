<?php
	include('shead.php');
	include('dbconnect.php');

?>
<style>
  body {
    font-family: 'Arial', sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  table {
    border-collapse: collapse;
    width: 80%;
    margin: 20px auto;
    background-color: #fff;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 50px;
  }

  th, td {
    padding: 15px;
    text-align: center;
    border-bottom: 1px solid #ddd;
  }

  tr:nth-child(even) {
    background-color: #f9f9f9;
  }

  input[type="submit"] {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
  }

  input[type="submit"]:hover {
    background-color: #45a049;
  }
</style>
<form method="POST" action="">

 <table border='2'>
                    <tr>
                        <td>Name</td>
                        <td>Product ID</td>
                        <!-- <td>Quantity</td> -->
                        <td>Price</td>
                        <td>Address</td>
                        <td>Phone</td>
                        <!-- <td>Status</td> -->
                        <td>Status</td>
                        <td>Deliver Status</td>
                    </tr>
<?php
session_start();
 $uname=$_SESSION['email']; 
                  $q="SELECT * fROM purchase where shop='$uname' and status='0'||status='2'||status='3'";
                  $m=mysql_query($q,$conn);
                  while($r=mysql_fetch_assoc($m))
                    {
                      $name=$r['name'];
                      $user=$r['user'];
                      $shop=$r['shop'];
                      $address=$r['address'];
                      $phone=$r['phone'];
                      $product_id=$r['product_id'];
                    //   $quantity=$r['quantity'];
                      $price=$r['price'];
                      $status=$r['status'];
?>
                    <tr>
                      
                        <td><?php echo $name; ?></td>
                        <td><?php echo $product_id; ?></td>
                        <!-- <td><?php echo $quantity; ?></td> -->
                        <td><?php echo $price; ?></td>
                        <td><?php echo $address; ?></td>
                        <td><?php echo $phone; ?></td>
                        <td><?php
                        if($status==0)
                          echo 'Ordered';
                        else if($status==2)
                          echo 'Order Cancelled';
                        else if($status==1)
                          echo 'Delivered';
                        else if($status==3)
                          echo 'Out for Delivery';
                        ?>
                      </td>
                      <td><input type="submit" name="submited" value="Out For Delivery"></td>
                      <td><input type="submit" name="submit" value="Delivered"></td></tr>
                  
<?php
}
?>
</table></form>
<?php

if(isset($_POST['submit'])){
 
  $q="update purchase set status='1' where shop='$shop' and user='$user' and product_id='$product_id'";
  // echo $q;
  if(mysql_query($q,$conn))
  {
    
          echo "<script>alert('Updated!');
              location.href='sorders.php';
          </script>";
        }
  }


?>

<?php

if(isset($_POST['submited'])){
  $q="update purchase set status='3' where shop='$shop' and user='$user' and product_id='$product_id'";
  // echo $q;
  if(mysql_query($q,$conn))
  {
          echo "<script>alert('Updated!');
              location.href='sorders.php';
          </script>";
        }
  }


?>