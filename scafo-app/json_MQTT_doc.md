
### JSON Message Documentation

- **sensorID**: A string representing the ID of the sensor. In this case, it's "dev000".

- **timestamp**: An integer representing the Unix timestamp (number of seconds since January 1, 1970) 
                of when the data was recorded. In this case, it's 1690296589.

- **status**: A string representing the status of the device. In this case, it can be: 
  - "0" : means device doesnot working 
  - "1" : means device is working

- **comment**: A string field for any additional comments related to the device. In this example, it's "... eventuali commenti ...".

- **Deformation_results**: An object containing deformation-related results.

  - **threshold**: A numeric value representing a threshold for the deformation. In this case, it's 0.001.

  - **status**: A string representing the status of the deformation results. In this case,it can be: 
    - "0" : there is no deformation
    - "1" : deformation detected

  - **Deformation Plot**: A list containing one element which is a base64 encoded image related to deformation.

- **shifting_results**: An object containing shifting-related results.

  - **threshold**: A numeric value representing a threshold for the shifting. In this case, it's 10.

  - **status**: A string representing the status of the shifting results. In this case, it can be: 
    - "0" : there is no shifting
    - "1" : shifting detected

  - **Dipole_base_info**: An object containing information about the shifted dipoles. Each key represents the dipole ID (e.g., "1", "2", etc.) and the value is an array containing the following elements:
    - Two arrays representing the coordinates of the Blue LED in dipole x and y points.
    - Two arrays representing the coordinates of the Green LED indipole x and y points.
    - A numeric value representing some information related to the dipole.

- **Shifted_dipoles**: An object containing information about shifted dipoles.

  - Each key (e.g., "1", "2", etc.) represents a dipole ID, and the corresponding value is a string representing some measurement related to that dipole.

- **ROI**: A list containing one element, which is an object representing the region of interest (ROI) defined by its corner coordinates.

  - **top_left**: An array with two elements, representing the x and y coordinates of the top-left corner of the ROI.

  - **top_right**: An array with two elements, representing the x and y coordinates of the top-right corner of the ROI.

  - **down_left**: An array with two elements, representing the x and y coordinates of the down-left corner of the ROI.

  - **down_right**: An array with two elements, representing the x and y coordinates of the down-right corner of the ROI.

