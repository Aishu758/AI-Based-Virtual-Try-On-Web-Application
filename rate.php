<?php
include('uhead.php');
include('dbconnect.php');

session_start();
$uname = $_SESSION['email'];
$id = $_GET['id'];
?>

<style>
    center {
        margin-top: 20px;
    }

    .rating-box {
        border-collapse: collapse;
        width: 70%;
        margin-top: 20px;
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 50px;
    }

    .rating-box th {
        background-color: #333;
        color: white;
        font-weight: bold;
        text-transform: uppercase;
    }

    .rating-box th,
    .rating-box td {
        padding: 15px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }

    .rating-box tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    .text-box {
        width: 100%;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-sizing: border-box;
        font-size: 16px;
    }

    .button {
        background-color: #e44d26;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }

    .button:hover {
        background-color: #d03c20;
    }
</style>

<center>
    <form method="POST" action="">
        <div class="search-container"></div><br>
        <div class="rating-box">
            <!-- Your product details display goes here -->
            <?php
            $sql = "select * from product where id='$id'";
            $m = mysql_query($sql, $conn);

            while ($row = mysql_fetch_assoc($m)) {
                echo '<div class="product-card" onclick="openLightbox(\'' . $row['photo'] . '\')">';
                echo '<div class="product-details">';
                echo '<h2>' . $row['brand'] . ' ' . $row['model'] . '</h2>';
                echo '<p>Frame Color: ' . $row['frame_color'] . '</p>';
                echo '<p>Lens Color: ' . $row['lens_color'] . '</p>';
                echo '<p>Price: $' . $row['price'] . '</p>';
                echo '</div>';
                echo '<img src="' . $row['photo'] . '" alt="Product Image" class="product-image" height="250px" width="50px">';
                echo '<div class="product-buttons">';
                // echo '<button onclick="editProduct(\'' . $row['id'] . '\')">EDIT</button>';
                // echo '<a href="ubuy.php?id=' . $row["id"] . '"><h4 style="color:#50f55c">BUY NOW   </h4></a>';
                echo '</div>';
                echo '</div>';
            }
            ?>

            <!-- Stylish text box -->
            <textarea class="text-box" name="feed" placeholder="Write Your Review" required></textarea>

            <!-- Button -->
            <input type="submit" class="button" name="submit" value="Submit Review">
        </div>

        <?php
        if (isset($_POST['submit'])) {
            // Handle form submission logic
            $p_id = $id; // Assuming $id is the product ID
            $feed = $_POST['feed'];
        
            // Insert data into the feedback table
            $insertQuery = "INSERT INTO feedback (p_id, feed) VALUES ('$p_id', '$feed')";
        
            if (mysql_query($insertQuery, $conn)) {
                echo "<script>alert('Review submitted successfully!');</script>";
            } else {
                echo "<script>alert('Error submitting review.');</script>";
            }
        }
        ?>
    </form>
</center>
