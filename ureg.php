<?php
	include('head.php');
	include('dbconnect.php');

?>

<br><br>
<center>
	<b><u><h1 style="color: white">REGISTRATION</h1></u></b><br>
<form  enctype="multipart/form-data" method="post">
	<table>
		<tr>
        <td style="color: white;">Name</td><td><input type="text" name="fname" class="form-control" style="margin-bottom: 8px;margin-left: 8px" required></td>
		</tr>
		<tr>
			<td style="color: white;">Address</td><td><input type="text" name="address" class="form-control"  style="margin-bottom: 8px;margin-left: 8px" required></td>
		</tr>
		<!-- <tr>
			<td>User Type</td><td>
				<select name="user" id="user" class="form-control"  style="margin-bottom: 8px;margin-left: 8px">
				<option value="user">USER</option>
				
				 <option value="shop">INSTITUTION</option> 
			</select>
			</td>
		</tr> -->

		<tr>
			<td style="color: white;">Phone</td><td><input type="text" name="phno" class="form-control"  style="margin-bottom: 8px;margin-left: 8px" pattern="(?=.*[0-9]).{10}" title="10 digit phone number" required></td>
		</tr>
		<tr>
			<td style="color: white;">Email</td><td><input type="email" name="email" class="form-control"  style="margin-bottom: 8px;margin-left: 8px" required></td>
		</tr>
		<tr>
			<td style="color: white;">Password </td><td><input type="text" name="password" class="form-control" style="margin-bottom: 8px;margin-left: 8px>" 
					pattern="(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}" 
					title="Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters"required
					></td>
		</tr>

		<tr>
			<td style="color: white;">Picture:</td><td> <input type="file" name="photo" required></td>

		<tr>
			<td></td><td><input type="submit" name="submit" value="submit" class="btn btn-success" style="padding: 10px 25px"></td>
		</tr>

	</table>
	
</form>
</center>
</div>
</div>
</div>
</div>
</div>
</section>


<?php

if(isset($_POST['submit'])){

	$target_dir = "pic/";
    $imageFileType = strtolower(pathinfo($_FILES["photo"]["name"], PATHINFO_EXTENSION));
    $target_file = $target_dir . basename($_FILES["photo"]["name"]);

    if (move_uploaded_file($_FILES["photo"]["tmp_name"], $target_file)) {
        //echo "The file ". htmlspecialchars(basename( $_FILES["photo"]["name"])). " has been uploaded.";
   } else {
        echo "Sorry, there was an error uploading your file.";
    }

	$name=$_POST['fname'];
  $address=$_POST['address'];
	//$user=$_POST['user'];
	$phno=$_POST['phno'];
	$email=$_POST['email'];
	$password=$_POST['password'];

	$q="INSERT into login(email,password,pic,status,user) values('$email','$password','$target_file','1','user')";
	// echo $q;
	if(mysql_query($q,$conn))
	{
		$id=0;
			$q1="select lid from login where email='$email' AND password='$password'";
			//echo $q1;
			$m=mysql_query($q1,$conn);
			while ($r=mysql_fetch_array($m)) {
				$id=$r['lid'];
			}
			$q2="INSERT INTO customer(name, email, city, address, phno, loginid) values('$name','$email','$city','$address','$phno','$id')";
				$n=mysql_query($q2,$conn);
				if($m&&$n){
					echo "<script>alert('Your account has been created!');
							location.href='login.php';
					</script>";
				}
	}
	}


?>


