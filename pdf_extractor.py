import logging
import os
from datetime import datetime

from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_renditions_element_type import \
    ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType

# Initialize the logger
logging.basicConfig(level=logging.INFO)


# This sample illustrates how to extract Text, Table Elements Information from PDF along with renditions of Figure, Table elements.

class Extract_TextTable_Info_With_Tables_Renditions_From_PDF():

    def __init__(self, input_pdf_path: str, make_output_path: str, CLIENT_ID: str, CLIENT_SECRET: str):
        self.input_pdf_path = input_pdf_path
        self.make_output_path = make_output_path

        #Adobe credentials
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET

        

        try:
            file = open(self.input_pdf_path, 'rb')
            input_stream = file.read()
            file.close()

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=self.CLIENT_ID,
                client_secret=self.CLIENT_SECRET,
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
                elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES],
                table_structure_type=TableStructureType.CSV,
            )

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Creates an output stream and copy stream asset's content to it
            global output_file_path
            output_file_path = self.create_output_file_path(self)
            


            with open(output_file_path, "wb") as file:
                file.write(stream_asset.get_input_stream())



        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Exception encountered while executing operation: {e}')

    
    # Generates a string containing a directory structure and file name for the output file
    @staticmethod
    def create_output_file_path(self) -> str:
        input_pdf_filename = os.path.splitext(os.path.basename(self.input_pdf_path))[0]
        os.makedirs(self.make_output_path, exist_ok=True)
        now = datetime.now()
        # time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")

        output_zipped_file = f"{self.make_output_path}/{input_pdf_filename}.zip"

        return output_zipped_file
    
    def output_path(self):
        return f"{output_file_path}"    
    

# if __name__ == "__main__":
#     input_pdf = "D:\Sridhar\htmltopdf2024-07-31T13-05-46.pdf"
#     output="pahthththt"

#     extractor = Extract_TextTable_Info_With_Tables_Renditions_From_PDF(input_pdf_path=input_pdf,make_output_path=output,CLIENT_ID=CLIENT_ID,CLIENT_SECRET=CLIENT_SECRET)