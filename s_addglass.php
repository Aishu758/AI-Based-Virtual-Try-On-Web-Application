<?php
	include('shead.php');
	include('dbconnect.php');

?>


<style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
            text-align: center;
            margin: 20px;
        }

        h1 {
            color: #333;
        }

        form {
        max-width: 400px;
        margin: 20px auto;
        background-color: #fff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
        text-align: left;
    }

    label {
        display: inline-block;
        font-weight: bold;
    }

    input {
        width: calc(100% - 10px); /* Adjust width as needed */
        padding: 8px;
        box-sizing: border-box;
        border: 1px solid #ccc;
        border-radius: 4px;
    }

    button {
        background-color: #4CAF50;
        color: #fff;
        padding: 10px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }

    button:hover {
        background-color: #45a049;
    }
    </style>
 
<!-- <h1>Add New Glasses</h1> -->

<form  enctype="multipart/form-data" method="post">
    <!-- Brand -->
    <label for="brand">Brand: <input type="text" id="brand" name="brand" required></label>

    <!-- Model -->
    <label for="model">Model: <input type="text" id="model" name="model" required></label>

    <!-- Frame Color -->
    <label for="frame_color">Frame Color: <input type="text" id="frame_color" name="frame_color" required></label>

    <!-- Lens Color -->
    <label for="lens_color">Lens Color: <input type="text" id="lens_color" name="lens_color" required></label>

    <!-- Price -->
    <label for="price">Price: <input type="number" id="price" name="price" min="0" step="0.01" required></label>

    <label for="pic">Picture: <input type="file" name="photo" required>

    <!-- Submit Button -->
    <input type="submit" name="submit" value="Add Glasses">
</form>




<?php


session_start();
    $email=$_SESSION['email']; 
    //echo "$email";




if(isset($_POST['submit'])){
	 $target_dir = "product/";
    $imageFileType = strtolower(pathinfo($_FILES["photo"]["name"], PATHINFO_EXTENSION));
    $target_file = $target_dir . basename($_FILES["photo"]["name"]);

    if (move_uploaded_file($_FILES["photo"]["tmp_name"], $target_file)) {
        //echo "The file ". htmlspecialchars(basename( $_FILES["photo"]["name"])). " has been uploaded.";
   } else {
        echo "Sorry, there was an error uploading your file.";
    }

    $brand=$_POST['brand'];
	$model=$_POST['model'];
    $frame_color=$_POST['frame_color'];
    $lens_color=$_POST['lens_color'];
	$price=$_POST['price'];
	
 
	$q="INSERT into product(username,brand,model,frame_color,price,photo,lens_color) values('$email','$brand','$model','$frame_color','$price','$target_file','$lens_color')";
    //echo $q;
    if(mysql_query($q,$conn))
    {
        echo "<script>alert('Glass Added!');
                           
                    </script>";

    }
    else{
         echo "<script>alert('Can't Add Plot!');
                            location.href='saddglass.php';
                    </script>";
    }



}

                     ?>