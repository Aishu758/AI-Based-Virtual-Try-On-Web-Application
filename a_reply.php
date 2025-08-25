  <?php
  include('dbconnect.php');
  include('admin_nav.php');
?>
   <section id="main-content">
      <section class="wrapper">
        <h3><i class="fa fa-angle-right"></i> Complaints </h3>
        <div class="row mb">
          <!-- page start-->
          <div class="content-panel">
            <div class="adv-table">
     
   <section id="main-content">
      <section class="wrapper">
        <h2> REPLY  </h2>
        <div class="row mb">
        


   <div class="send-message">
      <div class="container">
        <div class="row">
            <div>.</div>
            <div>.</div>
          <!-- <div class="col-md-12">
            <div class="section-heading">
              <h2>Send us a Message</h2>
            </div>
          </div> -->
          <div class="col-md-8">
            <div class="contact-form">
              <form id="contact" action="" method="POST" leftmargin='25%'>
                <div class="row">
                  <div class="col-lg-12 col-md-12 col-sm-12" >
                    <fieldset >
                      
                  <!-- <div class="col-lg-12 col-md-12 col-sm-12">
                    <fieldset>
                      <input name="subject" type="text" class="form-control" id="subject" placeholder="Subject" required="">
                    </fieldset>
                  </div> -->
                  <div class="col-lg-12" >
                    <fieldset>
                      <textarea name="message" rows="6" class="form-control" id="message" placeholder="Your Message" required=""></textarea>
                    </fieldset>
                  </div>
                  <div class="col-lg-12">
                    <fieldset>
                      <button type="submit" id="form-submit" name="msg" class="filled-button">Send Message</button>
                    </fieldset>
                  </div>

                </div>
              </form>
              
            </div>
          </div>
          <!-- <div class="col-md-4">
            <img src="car/assets/images/team_01.jpg" class="img-fluid" alt="">

            <h5 class="text-center" style="margin-top: 15px;">John Doe</h5>
          </div> -->
        </div>
      </div>
    </div>



     <?php

$id=$_GET['id'];
if(isset($_POST['msg']))
{
       
      // $subject=$_POST['subject'];
      $message=$_POST['message'];
      $u="UPDATE contact SET status='1' where email='$id'";
      $q="INSERT INTO reply(senderid, message) VALUES('$id', '$message')";
      $m=mysql_query($q,$conn);
      $c=mysql_query($u,$conn);
      if($m)
      {
        echo"<script>alert('Message successfully sent');
          location.href='a_complaint.php';
      </script>";
      }
      

}


?>



              
      
      </div>
    </div>
