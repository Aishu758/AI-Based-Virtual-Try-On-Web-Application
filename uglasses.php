<?php
	include('uhead.php');
	include('dbconnect.php');

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

        .product-card:hover {
            transform: scale(1.05); /* Added a scale effect on hover */
        }

        .product-details {
            flex: 1;
            margin-right: 20px;
        }

        .product-image {
            max-width: 150px;
            max-height: 150px;
            border-radius: 4px;
            object-fit: cover;
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

.product-buttons button:hover {
    background-color: #45a049; /* Darker green on hover */
}
.product-rating {
    display: flex;
    margin-top: 10px;
}

.rating-star {
    color: #fdd835; /* Color of the star (yellow in this example) */
    font-size: 20px; /* Adjust the size of the star as needed */
}
    </style>

<div class="containers">
    <?php

session_start();
    //$email=$_SESSION['email']; 

    // Assuming you have a MySQL connection ($conn) and want to fetch products
    $result = mysql_query("SELECT * FROM product");

    while ($row = mysql_fetch_assoc($result)) {
        echo '<div class="product-card" onclick="openLightbox(\'' . $row['photo'] . '\')">';
        echo '<div class="product-details">';
        echo '<h2>' . $row['brand'] . ' ' . $row['model'] . '</h2>';
        echo '<p>Frame Color: ' . $row['frame_color'] . '</p>';
        echo '<p>Lens Color: ' . $row['lens_color'] . '</p>';
        echo '<p>Price: $' . $row['price'] . '</p>';
        echo '<div class="product-rating">';

        $productId = $row['id'];
        $feedbackResult = mysql_query("SELECT label FROM feedback WHERE p_id = '$productId' AND label IS NOT NULL AND label != ''");
        // $a="SELECT label FROM feedback WHERE p_id = '$productId';";
        // echo $a;
        $totalFeedbacks = 0;
        $positiveFeedbacks = 0;
        while ($feedbackRow = mysql_fetch_assoc($feedbackResult)) {
            $totalFeedbacks++;
            if ($feedbackRow['label'] == 'positive') {
                $positiveFeedbacks++; // Increment for positive feedback
            }
        }
        $ratingOutOf5 = ($totalFeedbacks > 0) ? round(($positiveFeedbacks / $totalFeedbacks) * 5) : 0;
        for ($i = 1; $i <= 5; $i++) {
            if ($i <= floor($ratingOutOf5)) {
                echo '<span class="rating-star">&#9733;</span>'; // Colored yellow
            } else if ($i == ceil($ratingOutOf5) && $i <= 5 && $ratingOutOf5 - floor($ratingOutOf5) > 0) {
                echo '<span class="rating-star">&#9733;&#9734;</span>'; // Half-filled star
            } else {
                echo '<span class="rating-star">&#9734;</span>'; // Hollow star
            }
        }
        echo '<p style="font-size: small;">out of ' . $totalFeedbacks . ' ratings</p>';
echo '</div>';
        echo '</div>';
        
        echo '<img src="' . $row['photo'] . '" alt="Product Image" class="product-image">';
        
        echo '<div class="product-buttons">';
        
        // echo '<button onclick="editProduct(\'' . $row['id'] . '\')">EDIT</button>';
        echo '<a href="ubuy.php?id=' . $row["id"] . '"><h4 style="color:#50f55c">BUY NOW   </h4></a>';
        echo '</div>';
        echo '</div>';
    }
    
    echo '</div>';
    ?>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <img id="lightbox-image" style="width: 50%; height: 50%;">
</div>


<script>
   function openLightbox(clickedImageUrl) {
    var lightbox = document.getElementById('lightbox');
    lightbox.innerHTML = ''; // Clear the previous content

    var lightboxImage = document.createElement('img');
    lightboxImage.src = 'pic/th.jpg';
    lightboxImage.style.width = '50%';
    lightboxImage.style.height = '50%';
    lightbox.appendChild(lightboxImage);

    var clickedImage = new Image();
    clickedImage.src = clickedImageUrl;
    clickedImage.style.position = 'absolute';
    clickedImage.style.width = '15%'; //  of the first image's size
    clickedImage.style.height = '15%';
    clickedImage.style.marginTop = '-10px'; // Adjust the margin-top as needed

    lightbox.appendChild(clickedImage);
    lightbox.style.display = 'flex';
}




    function closeLightbox() {
        document.getElementById('lightbox').style.display = 'none';
    }

    function deleteProduct(id) {
        if (confirm("Are you sure you want to delete this product?")) {
            // Use AJAX to send a request to the server for deletion
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function () {
                if (xhr.readyState == 4 && xhr.status == 200) {
                    var response = JSON.parse(xhr.responseText);
                    if (response.success) {
                        alert("Product deleted successfully!");
                        location.reload(); // Refresh the page after deletion
                    } else {
                        alert("Failed to delete product!");
                    }
                }
            };
            xhr.open("GET", "delete_product.php?id=" + id, true);
            xhr.send();
        }
    }

    function editProduct(productId) {
        // Implement your edit logic here, you can redirect to an edit page or show a modal
        alert('Edit product with ID: ' + productId);
    }
</script>
