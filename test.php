<?php
include('uhead.php');
include('dbconnect.php');

session_start();
$uname = $_SESSION['email'];

if(isset($_POST['submit'])) {
   
}
?>

<style>
    table {
        border-collapse: collapse;
        width: 80%;
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
        <table border="2">
            <tr>
                <td>Product</td>
                <td>Status</td>
                <td>Price</td>
                <td>Action</td>
                <td>Rate</td>
            </tr>

            <?php
            $sql = "SELECT * FROM purchase WHERE user='$uname' AND status!='4'";
            $result = mysql_query($sql, $conn);

            while($row = mysql_fetch_array($result)) {
                $shop = $row['shop'];
                $product = $row['product_id'];
                $price = $row['price'];
                $status = $row['status'];

                $sql1 = "SELECT * FROM product WHERE id='$product'";
                $productDetails = mysql_fetch_array(mysql_query($sql1, $conn));
                $pic = $productDetails['photo'];
                ?>

                <tr>
                    <td><img src="<?php echo $pic ?>" height='20px' width='15px'></td>
                    <td><?php
                        if($status == 0)
                            echo 'Not Delivered';
                        else if($status == 1)
                            echo 'Delivered';
                        else if($status == 3)
                            echo 'Out For Delivery';
                        ?>
                    </td>
                    <td><?php echo $price ?></td>

                    <td>
                        <input type="submit" name="submit" value="Cancel Order">
                    </td>
                    <td>
                    <div class="rating">
                        <input type="radio" id="star5" name="rating" value="5">
                        <label for="star5" title="5 stars">&#9733;</label>
                        <input type="radio" id="star4" name="rating" value="4">
                        <label for="star4" title="4 stars">&#9733;</label>
                        <input type="radio" id="star3" name="rating" value="3">
                        <label for="star3" title="3 stars">&#9733;</label>
                        <input type="radio" id="star2" name="rating" value="2">
                        <label for="star2" title="2 stars">&#9733;</label>
                        <input type="radio" id="star1" name="rating" value="1">
                        <label for="star1" title="1 star">&#9733;</label>
                    </div>
                        <br> <!-- Add a line break for better separation -->
                        <!-- <input type="submit" name="submit" value="Rate"> -->
                    </td>
                </tr>

                <?php
            }
            ?>
        </table>
    </form>
</center>

<script>
    document.addEventListener('DOMContentLoaded', function () {
        var stars = document.querySelectorAll('.rating input[type="radio"] + label');

        stars.forEach(function (starLabel, index) {
            starLabel.addEventListener('click', function () {
                var clickedIndex = index + 1; // Adding 1 to convert from 0-based index to 1-based index

                for (var i = 1; i <= clickedIndex; i++) {
                    var star = document.getElementById('star' + i);
                    star.checked = true;
                    stars[i - 1].style.color = 'gold';
                }

                for (var j = clickedIndex + 1; j <= 5; j++) {
                    var unselectedStar = document.getElementById('star' + j);
                    unselectedStar.checked = false;
                    stars[j - 1].style.color = '#ddd';
                }
            });
        });
    });
</script>

