from langchain.document_loaders import DirectoryLoader, PyPDFLoader
path_folder='PDFs/'
raw_documents = DirectoryLoader(path_folder,
                                glob="**/*.pdf",
                                loader_cls=PyPDFLoader,
                                show_progress=True,
                                use_multithreading=True).load()
print(raw_documents)
'''-----------------------------------------------------'''
loader = DirectoryLoader('PDFs', use_multithreading=True, silent_errors=True,loader_cls=PyPDFLoader).load()
print(loader)