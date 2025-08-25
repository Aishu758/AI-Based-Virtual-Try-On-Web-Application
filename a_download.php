<?php

  include_once('dbconnect.php');
?>



<?php

header('Content-Type: text/csv; charset=utf-8');
      header('Content-Disposition: attachment; filename=testdata.csv');
       $output = fopen("php://output", "w");
     fputcsv($output, array('pid','inputstring'));
         $query = "select *  from feedback" ;
         $result = mysql_query($query, $conn);
         while($row = mysql_fetch_array($result))
         {
              fputcsv($output,array($row['id'],$row['feed']));
             
         }
      fclose($output);

?>