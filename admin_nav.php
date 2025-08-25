<head>
<title> Admin Panel</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="keywords" content="Visitors Responsive web template, Bootstrap Web Templates, Flat Web Templates, Android Compatible web template, 
Smartphone Compatible web template, free webdesigns for Nokia, Samsung, LG, SonyEricsson, Motorola web design" />
<script type="application/x-javascript"> addEventListener("load", function() { setTimeout(hideURLbar, 0); }, false); function hideURLbar(){ window.scrollTo(0,1); } </script>
<!-- bootstrap-css -->
<link rel="stylesheet" href="admin/admin_temp/css/bootstrap.min.css" >
<!-- //bootstrap-css -->
<!-- Custom CSS -->
<link href="admin/admin_temp/css/style.css" rel='stylesheet' type='text/css' />
<link href="admin/admin_temp/css/style-responsive.css" rel="stylesheet"/>
<!-- font CSS -->
<link href='//fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900,900italic' rel='stylesheet' type='text/css'>
<!-- font-awesome icons -->
<link rel="stylesheet" href="admin/admin_temp/css/font.css" type="text/css"/>
<link href="admin/admin_temp/css/font-awesome.css" rel="stylesheet"> 
<link rel="stylesheet" href="admin/admin_temp/css/morris.css" type="text/css"/>
<!-- calendar -->
<link rel="stylesheet" href="admin/admin_temp/css/monthly.css">
<!-- //calendar -->
<!-- //font-awesome icons -->
<script src="admin/admin_temp/js/jquery2.0.3.min.js"></script>
<script src="admin/admin_temp/js/raphael-min.js"></script>
<script src="admin/admin_temp/js/morris.js"></script>
</head>
<body>
<section id="container">
<!--header start-->
<header class="header fixed-top clearfix">
<!--logo start-->
<div class="brand">
    <a href="index.php" class="logo">
        Admin
    </a>
    <div class="sidebar-toggle-box">
        <div class="fa fa-bars"></div>
    </div>
</div>

<div class="top-nav clearfix">
    <!--search & user info start-->
    <ul class="nav pull-right top-menu">

        <li class="dropdown">
            <a data-toggle="dropdown" class="dropdown-toggle" href="#">
                <img alt="" src="admin/admin_temp/images/2.png">
                <span class="username">Admin</span>
                <b class="caret"></b>
            </a>
            <ul class="dropdown-menu extended logout">
             
                <li><a href="./index.html"><i class="fa fa-key"></i> Log Out</a></li>
            </ul>
        </li>
        
       
    </ul>
   
</div>
</header>
<!--header end-->
<!--sidebar start-->
<aside>
    <div id="sidebar" class="nav-collapse">
        <!-- sidebar menu start-->
        <div class="leftside-navigation">
            <ul class="sidebar-menu" id="nav-accordion">
                <li>
                    <a class="active" href="a_home.php">
                        <i class="fa fa-dashboard"></i>
                        <span>Dashboard</span>
                    </a>
                </li>
                
               <li class="sub-menu">
            <a href="a_profile.php">
              <i class="fa fa-user"></i>
              <span>My Profile</span>
              </a>

          </li> 
          <li class="sub-menu">
            <a href="a_ureq.php">
              <i class="fa fa-user"></i>
              <span>New Users</span>
              </a>

          </li> 
          <li class="sub-menu">
            <a href="a_uapproved.php">
              <i class="fa fa-user"></i>
              <span>Approved Shops</span>
              </a>

          </li> 
          <li class="sub-menu">
            <a href="a_urejected.php">
              <i class="fa fa-user"></i>
              <span>Rejected Shops</span>
              </a>

          </li> 
          <li class="sub-menu">
            <a href="a_download.php">
              <i class="fa fa-download"></i>
              <span>Download Test Data</span>
              </a>
          </li> 
          <li class="sub-menu">
            <a href="a_upload.php">
              <i class="fa fa-upload"></i>
              <span>Upload Result</span>
              </a>
          </li> 
          <!-- <li class="sub-menu">
            <a href="add.php">
              <i class="fa fa-building"></i>
              <span>Add</span>
              </a>

          </li>  -->
          <li class="sub-menu">
            <a href="a_complaint.php">
              <i class="fa fa-envelope"></i>
              <span>Complaint</span>
              </a>

          </li>   
                
          <li class="sub-menu">
            <a href="a_view.php">
              <i class="fa fa-envelope"></i>
              <span>Feedbacks</span>
              </a>

          </li>   
                
               
            </ul>            </div>
        <!-- sidebar menu end-->
    </div>
</aside>

</section>
   
    <script src="admin/admin_temp/js/bootstrap.js"></script>
    <script src="admin/admin_temp/js/jquery.dcjqaccordion.2.7.js"></script>
    <script src="admin/admin_temp/js/scripts.js"></script>
    <script src="admin/admin_temp/js/jquery.slimscroll.js"></script>
    <script src="admin/admin_temp/js/jquery.nicescroll.js"></script>
<!--[if lte IE 8]><script language="javascript" type="text/javascript" src="js/flot-chart/excanvas.min.js"></script><![endif]-->
    <script src="admin/admin_temp/js/jquery.scrollTo.js"></script>
<!-- morris JavaScript -->	
<script>
	$(document).ready(function() {
		//BOX BUTTON SHOW AND CLOSE
	   jQuery('.small-graph-box').hover(function() {
		  jQuery(this).find('.box-button').fadeIn('fast');
	   }, function() {
		  jQuery(this).find('.box-button').fadeOut('fast');
	   });
	   jQuery('.small-graph-box .box-close').click(function() {
		  jQuery(this).closest('.small-graph-box').fadeOut(200);
		  return false;
	   });
	   

//	</script>
<!-- calendar -->
<script type="text/javascript" src="../js/monthly.js"></script>
	<script type="text/javascript">//
//		
//	</script>
	<!-- //calendar -->
        
       
</body>
</html>
