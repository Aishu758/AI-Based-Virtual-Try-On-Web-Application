<?php
	include('uhead.php');
	include('dbconnect.php');

    session_start();
    $uname=$_SESSION['email']; 
$id=$_GET['id'];


?>



<style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        header {
            background-color: #333;
            color: #fff;
            text-align: center;
            padding: 10px;
        }

        .containers {
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        .product-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            cursor: pointer;
            transition: transform 0.3s; /* Added a smooth transition on hover */
        }

        

        .product-details {
            flex: 1;
            margin-right: 10px;
        }

        .product-image {
            max-width: 150px;
            max-height: 150px;
            border-radius: 4px;
            object-fit: cover;
            margin-right: 50px;
        }

        .lightbox {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            justify-content: center;
            align-items: center;
            z-index: 999; /* Ensure the lightbox is on top of other elements */
        }

        .lightbox img {
            max-width: 80%;
            max-height: 80%;
            border-radius: 4px;
        }
        .product-buttons {
    position: absolute;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
    opacity: 0; /* Initially hidden */
    transition: opacity 0.3s;
}

.product-card:hover .product-buttons {
    opacity: 1; /* Visible on hover */
}

.product-buttons button {
    background-color: #4caf50; /* Green color for buttons */
    color: #fff;
    padding: 8px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
}
table {
    border-collapse: collapse;
    width: 400px;
    background-color: #fff;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
  }
  tr, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
  }
  input[type="text"] {
    width: 100%;
    padding: 8px;
    margin: 5px 0;
    box-sizing: border-box;
    border: 1px solid #ccc;
    border-radius: 4px;
  }
  input[type="submit"] {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  input[type="submit"]:hover {
    background-color: #45a049;
  }
.product-buttons button:hover {
    background-color: #45a049; /* Darker green on hover */
}
    </style>

<div class="containers">
    <?php

    // Assuming you have a MySQL connection ($conn) and want to fetch products
    $result = mysql_query("SELECT * fROM product where id=$id");

    while ($row = mysql_fetch_assoc($result)) {
        $price=$row['price'];
        $shop=$row['username'];

        echo '<div class="product-card" onclick="openLightbox(\'' . $row['photo'] . '\')">';
        echo '<div class="product-details">';
        echo '<h2>' . $row['brand'] . ' ' . $row['model'] . '</h2>';
        echo '<p>Frame Color: ' . $row['frame_color'] . '</p>';
        echo '<p>Lens Color: ' . $row['lens_color'] . '</p>';
        echo '<p>Price: $' . $row['price'] . '</p>';
        echo '</div>';
        echo '<img src="' . $row['photo'] . '" alt="Product Image" class="product-image">';
       ?>
<br>
<form action="" method="POST">
                    <center>
                    <table>
                    <!-- <tr><td>Quantity:</td><td><input type="number" name="quantity" placeholder="Quantity in kg" required></td></tr> -->
                    <tr><td>Name:</td><td><input type="text" name="name" placeholder="Your Name" pattern="[a-zA-Z]{3,30}" title="format" required></td></tr>
                    <tr><td>Address:</td><td><input type="text" name="address" placeholder="Address to deliver" required></td></tr>
                    <tr><td>Phone:</td><td><input type="text" name="phone" placeholder="Contact Number" pattern="[0-9]{10}" title="10 digit phone number" required></td></tr>
                    <tr><td>Price:</td><td><input type="text" name="price" placeholder="<?php echo $row['price']; ?>" readonly></td></tr>
                     <tr><td><input type='submit' name="submit" value="BUY"></td></tr>
                 </table>
                 </center>
                  </form>

                  <?php

echo '<div class="product-buttons">';

        echo '</div>';
        echo '</div>';
    }
    
    echo '</div>';


    if(isset($_POST['submit'])){

        //$quantity=$_POST['quantity'];
        $name=$_POST['name'];
        //$id=$_POST['name'];
        //$price=$_POST['price'];
        $address=$_POST['address'];
        $phone=$_POST['phone'];
        //$price1=$price*$quantity;
        //$price2=($price1*10)/100;
        
        $q="INSERT into purchase(name,user,shop,address,phone,product_id,price) values('$name','$uname','$shop','$address','$phone','$id','$price')";
        $q1="select id from purchase order by id desc limit 1";
                  //echo $q1;
                  $m=mysql_query($q1,$conn);
                  while ($r=mysql_fetch_array($m)) {
                      $id=$r['id'];
                  }
       
        // echo $q;
        if(mysql_query($q,$conn))
        {
          
                echo "<script>alert('Product Ordered!');
                    location.href='uglasses.php';
                </script>";
              }
        }
      
      
      ?>