<?php
include('dbconnect.php');
include('admin_nav.php');
?>

<style>
    .wrapper {
        margin: 80px 0 0 20px;
    }

    h3 {
        color: #2e9fff;
    }

    .content-panel {
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        padding: 20px;
        border-radius: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }

    .message-box {
        border: 1px solid #ccc;
        padding: 15px;
        margin-bottom: 20px;
        width: calc(50% - 20px); /* Set the width to 50% of the container with some gap */
        background-color: #fff;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
    }

    p {
        margin: 0;
    }
</style>

<section id="main-content">
    <div class="wrapper">
        <h3><i class="fa fa-angle-right"></i> Feedback </h3><br>
        <div class="content-panel">
            <?php
            $q = "SELECT * FROM contact";
            $m = mysql_query($q, $conn);

            while ($r = mysql_fetch_assoc($m)) {
            ?>
                <div class="message-box">
                    <p><strong>Name:</strong> <?php echo $r['name']; ?></p>
                    <p><strong>Email:</strong> <?php echo $r['email']; ?></p>
                    <p><strong>Message:</strong> <?php echo $r['message']; ?></p>
                    <p><strong>Label:</strong> <?php echo $r['label']; ?></p>
                </div>
            <?php
            }
            ?>
        </div>
    </div>
</section>
