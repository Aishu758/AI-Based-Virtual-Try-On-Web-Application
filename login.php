<?php
	include('head.php');
	include('dbconnect.php');

?>

<br><br>
<center>
<b><h1 style="color: white">Login</h1></b><br>
	<form method="POST">
		<table>
			<tr>
                <td  style="color: white;" style="margin-right: 30px">Email :</td><td><input type="email" name="email" class="form-control" style="margin-bottom: 20px; margin-left: 20px" pattern="[a-z0-9._%=-]+@[a-z0-9.-]+\.[a-z]{2,}$"></td>
            </tr>
            <tr>
                <td  style="color: white;" style="margin-right: 30px">Password :</td><td><input type="password" name="password" class="form-control" style="margin-bottom: 20px; margin-left: 20px" required></td>

            </tr>
            <tr>
			<td></td><td><input type="submit" name="submit" value="submit" class="btn btn-success" style="padding: 10px 25px"></td>
		</tr>
		</table>
	</form>
</center>
</section>
</div>
<?php
if(isset($_POST['submit']))
{
	$email=$_POST['email'];
	$password=$_POST['password'];
	$user=0;
	$id=0;

	$q="SELECT * from login WHERE email='$email' and password='$password' ";
	 //echo $q;
	 //break;
	 $m=mysql_query($q,$conn);
	if($m)
	{
	
		while($r=mysql_fetch_array($m)){
				$id=$r['lid'];
				$user=$r['user'];
				$status=$r['status'];
				$email=$r['email'];
				 if($status==1)
	       {
	        $flag=1;
	  	    session_start();
	        
	  	    $_SESSION['user'] = $user;
	  	    $_SESSION['email'] = $email;
	  	    $_SESSION['lkey'] = $id;
	  	   // $_SESSION[$id];
	      }
			

			}
	
			
				
			
			// echo $user;
		if( $user=='user' && $status=='1' ){
				echo "<script>
					location.href='uhome.php';
				</script> ";
			}
			else if( $user=='shop' && $status=='1' ){
				echo "<script>
					location.href='shome.php';
				</script> ";
			}
			else if( $user=='institution' && $status=='1' ){
				echo "<script>
					location.href='ihome.php';
				</script> ";
			}
		else if( $status=='3'){
				echo "<script>
				location.href='a_home.php';
				</script>";
			}
	

		else{
				echo "<script> alert('user not registered ');
				location.href='login.php';
				</script>";
}
}
}

?>