<?php
	include('uhead.php');
	include('dbconnect.php');

?>

<style>


  center {
    margin-top: 20px;
  }

  table {
    border-collapse: collapse;
    width: 90%;
    margin-top: 20px;
    background-color: #fff;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 50px;
  }
   th {
    background-color: #333;
    color: white;
    font-weight: bold;
    text-transform: uppercase;
  }

  th, td {
    padding: 15px;
    text-align: center;
    border-bottom: 1px solid #ddd;
  }

  tr:nth-child(even) {
    background-color: #f9f9f9;
  }
  .rating {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    input[type="radio"] {
        display: none;
    }

    label {
        font-size: 25px;
        color: #ddd;
        cursor: pointer;
        transition: color 0.3s;
    }
    input[type="submit"] {
        background-color: #e44d26;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    input[type="submit"]:hover {
        background-color: #d03c20;
    }

    input[type="radio"]:checked + label {
        color: gold;
    }
 
</style>

<center> 
<form method="POST" action="">
    <div class="search-container">
    
  </div>
   <table border="2">
    <tr>
      <!-- <td>Product</td> -->
      <!-- <td>Quantity</td> -->
      <td>Price</td>
      <td>Product</td>
      <td>Status</td>
      <td>Action</td>
      <td>Rate</td>
    </tr>
<?php
session_start();
              $uname=$_SESSION['email']; 
  $sql="select * from purchase where user='$uname' and status!='4'";
                  $m=mysql_query($sql,$conn);
                           
                  while($r=mysql_fetch_array($m))
                    {
                  $shop=$r['shop'];
                  $product=$r['product_id'];
                  //$quantity=$r['quantity'];
                  $price=$r['price'];
                  $status=$r['status'];
                  $sql1 = "SELECT * FROM product WHERE id='$product'";
                  $rr = mysql_fetch_array(mysql_query($sql1, $conn));
                  $pic = $rr['photo'];
  echo $pic;



                  ?>
                 

                    <tr>
                      <!-- <td><?php echo $product ?></td> -->
                      <!-- <td><?php echo $quantity ?></td> -->
                      <td><?php echo $price ?></td>
                      <td><img src="<?php echo $pic ?>" height='20px' width='15px'></td>
                      <td><?php 
                            if($status==0)
                              echo 'Not Delivered';
                            else if($status==1)
                              echo 'Delivered';
                            else if($status==3)
                              echo 'Out For Delivery';

                       ?></td>
                     <td><input type="submit" name="submit" value="Cancel Order"></td>
                     <td><a href="rate.php?id=<?php echo $product ?>" style="color:#50f55c">RATE</a></td>

                     
                    </td></tr>
                


                  <?php
                } 

?>
  </table>
 
<?php

if(isset($_POST['submit'])){
 
  $q="update purchase set status='4' where user='$uname' and shop='$shop' and product='$product'";
  // echo $q;
  if(mysql_query($q,$conn))
  {
          echo "<script>alert('Item Cancelled!');
              location.href='uorder.php';
          </script>";
        }
  }

  // if(isset($_POST['submit1'])){
  //   <a href="ubuy.php?id=' . $row["id"] . '"><h4 style="color:#50f55c">BUY NOW   </h4></a>
  // }
 
    
?>