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
           
     <?php
                  $q="SELECT * fROM contact where status=0";
                  //echo $q;
                  $m=mysql_query($q,$conn);
                  // if($m){
                  //  echo "good";
                  // }
                  
                  while($r=mysql_fetch_assoc($m))
                    {

?>

        
              <div class="col-md-6">
                <div class="product-item">
                  <?php $name=$r['name']?>
                  <!-- <?php $sub=$r['subject']?> -->
                  <?php $msg=$r['message']?>
                  <!-- <a href="car-details.php?id=<?php echo $key?>"> -->
                    


               <div class="col-sm-12">
                    <!-- <label class="text-dark font-weight-medium" for="">Name</label> -->
                    <div class="input-group mb-2">
Name      :
                         <input type="text" class="form-control" id="hname" name="hname" value="<?php echo $r['name'];?>" 
                         disabled="disabled">
                    </div>
               </div>
                    
               <div class="col-sm-12">
                   
                    <div class="input-group mb-2">
E mail
                         <input type="text" class="form-control" id="email" name="email" value="<?php echo $r['email'];?>" 
                         disabled="disabled">
                    </div>
               </div>
               
               <div class="col-sm-12">
                    <!-- <label class="text-dark font-weight-medium" for="">Subject</label> -->
                    <div class="input-group mb-2">
<!-- Subject
                         <input type="text" class="form-control" id="sub" name="sub" value="<?php echo $r['subject'];?>" 
                         disabled="disabled">
                    </div>
               </div> 

               <div class="col-sm-12">
                    <label class="text-dark font-weight-medium" for="">Message</label>
                    <div class="input-group mb-2">-->
Message
                         <input type="text" class="form-control" id="msg" name="msg" value="<?php echo $r['message'];?>" 
                         disabled="disabled">
                    
               </div>
                </div>
                <a href="a_reply.php?id=<?php echo $r['email'] ?>">
               <button class="my-1 btn btn-sm btn-success" name="submit" type="submit">REPLY</button>
          </a>
              </div>
            </div>
</div>
           <?php

             }
             ?>

              
          </div>
        </div>
      </div>
    </div>
