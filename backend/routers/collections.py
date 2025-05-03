import os
from fastapi import APIRouter, Depends, HTTPException, status, Query, File, Form, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
import json # Needed for ingest-file params
from typing import List, Dict, Any # Needed for background tasks and helper

# Database imports
from database.connection import get_db, get_chroma_client
from database.service import CollectionService # Assuming this is the correct location
from database.models import Collection # Import Collection model
from database.models import FileRegistry, FileStatus # Needed for ingest background tasks

# Schema imports
from schemas.collection import (
    CollectionCreate, 
    CollectionResponse, 
    CollectionList,
    CollectionCreateResponse,
    EmbeddingsModel
)
from schemas.ingestion import (
    IngestURLRequest, 
    AddDocumentsRequest, 
    AddDocumentsResponse
)
from schemas.query import QueryRequest, QueryResponse
from schemas.files import FileRegistryResponse # Assuming schemas/files.py exists or will be created

# Service imports
from services.collections import CollectionsService
from services.ingestion import IngestionService
from services.query import QueryService

# Dependency imports
from dependencies import verify_token

router = APIRouter(
    prefix="/collections",
    tags=["Collections"],
    dependencies=[Depends(verify_token)]
)

# Helper function to get and validate collection existence in both databases
def _get_and_validate_collection(db: Session, collection_id: int):
    """
    Retrieves a collection by ID from SQLite and validates its existence in ChromaDB.
    
    Args:
        db: SQLAlchemy Session
        collection_id: ID of the collection
        
    Returns:
        Tuple: (collection_object, collection_name)
        
    Raises:
        HTTPException: If collection not found in either database.
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
    
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
        
    return collection, collection_name

# Create a new collection (handle both POST /collections and /collections/)
@router.post(
    "", # Route without trailing slash
    response_model=CollectionResponse, # Reverted: Return full response
    summary="Create collection (no slash)",
    include_in_schema=False, # Hide from OpenAPI docs
    status_code=status.HTTP_201_CREATED # Explicitly set 201 status
)
@router.post(
    "/", # Route with trailing slash
    response_model=CollectionResponse, # Reverted: Return full response
    summary="Create collection",
    description="""Create a new knowledge base collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections' \ 
      -H 'Authorization: Bearer 0p3n-w3bu!' \ 
      -H 'Content-Type: application/json' \ 
      -d '{
        "name": "my-knowledge-base",
        "description": "My first knowledge base",
        "owner": "user1",
        "visibility": "private",
        "embeddings_model": {
          "model": "default",
          "vendor": "default",
          "endpoint":"default",
          "apikey": "default"
        }
        
        # For OpenAI embeddings, use:
        # "embeddings_model": {
        #   "model": "text-embedding-3-small",
        #   "vendor": "openai",
        #   "endpoint":"https://api.openai.com/v1/embeddings"
        #   "apikey": "your-openai-key-here"
        # }
    '
    ```
    """,
    responses={
        201: {"description": "Collection created successfully"},
        400: {"description": "Bad request - Invalid collection data"},
        409: {"description": "Conflict - Collection with this name already exists"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    },
    status_code=status.HTTP_201_CREATED
)
async def create_collection(
    collection: CollectionCreate, 
    db: Session = Depends(get_db)
):
    """Create a new knowledge base collection.
    
    Args:
        collection: Collection data from request body
        db: Database session
        
    Returns:
        The created collection
        
    Raises:
        HTTPException: If collection creation fails
    """
    # Resolve default values for embeddings_model before passing to service
    if collection.embeddings_model:
        model_info = collection.embeddings_model.model_dump()
        resolved_config = {}
        
        # Resolve vendor
        vendor = model_info.get("vendor")
        if vendor == "default":
            vendor = os.getenv("EMBEDDINGS_VENDOR")
            if not vendor:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_VENDOR environment variable not set but 'default' specified"
                )
        resolved_config["vendor"] = vendor
        
        # Resolve model
        model = model_info.get("model")
        if model == "default":
            model = os.getenv("EMBEDDINGS_MODEL")
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_MODEL environment variable not set but 'default' specified"
                )
        resolved_config["model"] = model
        
        # Resolve API key (optional)
        api_key = model_info.get("apikey")
        if api_key == "default":
            api_key = os.getenv("EMBEDDINGS_APIKEY", "")
        
        # Only log whether we have a key or not, never log the key itself or its contents
        if vendor == "openai":
            print(f"INFO: [router.create_collection] Using OpenAI API key: {'[PROVIDED]' if api_key else '[MISSING]'}")
            
        resolved_config["apikey"] = api_key
        
        # Resolve API endpoint (needed for some vendors like Ollama)
        api_endpoint = model_info.get("api_endpoint")
        if api_endpoint == "default":
            api_endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
            if not api_endpoint and vendor == "ollama":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="EMBEDDINGS_ENDPOINT environment variable not set but 'default' specified for Ollama"
                )
        if api_endpoint:  # Only add if not None
            resolved_config["api_endpoint"] = api_endpoint
            
        # Log the resolved configuration
        print(f"INFO: [router.create_collection] Resolved embeddings config: {resolved_config}")
        
        # Replace default values with resolved values in the collection object
        collection.embeddings_model = EmbeddingsModel(**resolved_config)
    
    # Now call the service with default values already resolved
    created_collection = CollectionsService.create_collection(collection, db)
    # Return the full collection object as defined by CollectionResponse
    return created_collection


# List collections (handle both /collections and /collections/)
@router.get(
    "", # Route without trailing slash
    response_model=CollectionList,
    summary="List collections (no slash)", # Indicate purpose
    include_in_schema=False # Hide from OpenAPI docs to avoid duplication
)
@router.get(
    "/", # Route with trailing slash
    response_model=CollectionList,
    summary="List collections",
    description="""List all available knowledge base collections.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    
    # With filtering parameters
    curl -X GET 'http://localhost:9090/collections?owner=user1&visibility=public&skip=0&limit=20' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Collections"],
    responses={
        200: {"description": "List of collections"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def list_collections(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of collections to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of collections to return"),
    owner: str = Query(None, description="Filter by owner"),
    visibility: str = Query(None, description="Filter by visibility ('private' or 'public')")
):
    """List all available knowledge base collections with optional filtering.
    
    Args:
        db: Database session
        skip: Number of collections to skip
        limit: Maximum number of collections to return
        owner: Optional filter by owner
        visibility: Optional filter by visibility
        
    Returns:
        List of collections matching the filter criteria
    """
    return CollectionsService.list_collections(
        db=db,
        skip=skip,
        limit=limit,
        owner=owner,
        visibility=visibility
    )


# Get a specific collection
@router.get(
    "/{collection_id}",
    response_model=CollectionResponse,
    summary="Get collection",
    description="""Get details of a specific knowledge base collection.
    
    Example:
    ```bash
    curl -X GET 'http://localhost:9090/collections/1' \
      -H 'Authorization: Bearer 0p3n-w3bu!'
    ```
    """,
    tags=["Collections"],
    responses={
        200: {"description": "Collection details"},
        404: {"description": "Not found - Collection not found"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"}
    }
)
async def get_collection(
    collection_id: int,
    db: Session = Depends(get_db)
):
    """Get details of a specific knowledge base collection.
    
    Args:
        collection_id: ID of the collection to retrieve
        db: Database session
        
    Returns:
        Collection details
        
    Raises:
        HTTPException: If collection not found
    """
    return CollectionsService.get_collection(collection_id, db)


# Ingestion endpoints nested under collections

@router.post(
    "/{collection_id}/ingest-url",
    response_model=AddDocumentsResponse,
    summary="Ingest content from URLs directly into a collection",
    description="""Fetch, process, and add content from URLs to a collection in one operation.

    This endpoint fetches content from specified URLs, processes it with the URL ingestion plugin,
    and adds the content to the collection.

    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-url' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "urls": ["https://example.com/page1", "https://example.com/page2"],
        "plugin_params": {
          "chunk_size": 1000,
          "chunk_unit": "char",
          "chunk_overlap": 200
        }
      }'
    ```

    Parameters for url_ingest plugin:
    - urls: List of URLs to ingest
    - chunk_size: Size of each chunk (default: 1000)
    - chunk_unit: Unit for chunking (char, word, line) (default: char)
    - chunk_overlap: Number of units to overlap between chunks (default: 200)
    """,
    tags=["Ingestion"], # Add Ingestion tag here
    responses={
        200: {"description": "URLs ingested successfully"},
        400: {"description": "Invalid plugin parameters or URLs"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or plugin not found"},
        500: {"description": "Error processing URLs or adding to collection"}
    }
)
async def ingest_url_to_collection(
    collection_id: int,
    request: IngestURLRequest,
    # token: str = Depends(verify_token), # Token verified by router dependency
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None # Add BackgroundTasks dependency
):
    """Ingest content from URLs directly into a collection.

    This endpoint fetches content from specified URLs, processes it with the URL ingestion plugin,
    and adds the content to the collection.

    Args:
        collection_id: ID of the collection
        request: Request with URLs and processing parameters
        # token: Authentication token # Removed
        db: Database session
        background_tasks: FastAPI background tasks
        
    Returns:
        Status information about the ingestion operation
        
    Raises:
        HTTPException: If collection not found, plugin not found, or ingestion fails
    """
    # Check if collection exists in SQLite
    collection = CollectionService.get_collection(db, collection_id)
    if not collection:
        raise HTTPException(
            status_code=404,
            detail=f"Collection with ID {collection_id} not found in database"
        )
        
    # Get collection name - handle both dict-like and attribute access
    collection_name = collection['name'] if isinstance(collection, dict) else collection.name
        
    # Also verify ChromaDB collection exists
    try:
        chroma_client = get_chroma_client()
        chroma_collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' exists in database but not in ChromaDB. Please recreate the collection."
        )
    
    # Get and validate the collection using the helper
    collection, collection_name = _get_and_validate_collection(db, collection_id)
    
    # Check if plugin exists
    plugin_name = request.plugin_name
    plugin = IngestionService.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion plugin '{plugin_name}' not found"
        )
    
    # Check if URLs are provided
    if not request.urls:
        raise HTTPException(
            status_code=400,
            detail="No URLs provided"
        )
    
    try:
        # Create a temporary file to track this URL ingestion
        import tempfile
        import uuid
        import os
        
        temp_dir = os.path.join(tempfile.gettempdir(), "url_ingestion")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.url")
        
        # Write URLs to the temporary file (just for tracking)
        with open(temp_file_path, "w") as f:
            for url in request.urls:
                f.write(f"{url}\n")
        
        # Step 1: Register the URL ingestion in the FileRegistry with PROCESSING status
        # Store the first URL as the filename to make it easier to display and preview
        first_url = request.urls[0] if request.urls else "unknown_url"
        file_registry = IngestionService.register_file(
            db=db,
            collection_id=collection_id,
            file_path=temp_file_path,
            file_url=first_url,  # Store the URL for direct access
            original_filename=first_url,  # Use the first URL as the original filename for better display
            plugin_name="url_ingest",  # Ensure consistent plugin name
            plugin_params={"urls": request.urls, **request.plugin_params},
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            document_count=0,  # Will be updated after processing
            content_type="text/plain",
            status=FileStatus.PROCESSING  # Set initial status to PROCESSING
        )
        
        # Step 2: Schedule background task for processing and adding documents
        def process_urls_in_background(urls: List[str], plugin_name: str, params: dict, 
                                   collection_id: int, file_registry_id: int):
            try:
                # Create a new session for the background task
                from database.connection import SessionLocal
                db_background = SessionLocal()
                
                try:
                    # Make a placeholder file path for the URL ingestion
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    
                    # Step 2.1: Process URLs with plugin
                    # Add URLs to the plugin parameters
                    full_params = {**params, "urls": urls}
                    
                    documents = IngestionService.ingest_file(
                        file_path=temp_file_path,  # This is just a placeholder
                        plugin_name=plugin_name,
                        plugin_params=full_params
                    )
                    
                    # Step 2.2: Add documents to collection
                    result = IngestionService.add_documents_to_collection(
                        db=db_background,
                        collection_id=collection_id,
                        documents=documents
                    )
                    
                    # Step 2.3: Update file registry with completed status and document count
                    IngestionService.update_file_status(
                        db=db_background, 
                        file_id=file_registry_id, 
                        status=FileStatus.COMPLETED
                    )
                    
                    # Update document count
                    file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                    if file_reg:
                        file_reg.document_count = len(documents)
                        db_background.commit()
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                    
                finally:
                    db_background.close()
                    
            except Exception as e:
                print(f"ERROR: [background_task] Failed to process URLs: {str(e)}")
                
                # Update file status to FAILED
                try:
                    from database.connection import SessionLocal
                    db_error = SessionLocal()
                    try:
                        IngestionService.update_file_status(
                            db=db_error, 
                            file_id=file_registry_id, 
                            status=FileStatus.FAILED
                        )
                    finally:
                        db_error.close()
                except Exception:
                    print(f"ERROR: [background_task] Could not update file status to FAILED")
        
        # Add the task to background tasks
        background_tasks.add_task(
            process_urls_in_background, 
            request.urls, 
            plugin_name, 
            request.plugin_params, 
            collection_id, 
            file_registry.id
        )
        
        # Return immediate response with URL information
        return {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "documents_added": 0,  # Initially 0 since processing will happen in background
            "success": True,
            "file_path": temp_file_path,
            "file_url": "",
            "original_filename": f"urls_{len(request.urls)}",
            "plugin_name": plugin_name,
            "file_registry_id": file_registry.id,
            "status": "processing"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest URLs: {str(e)}"
        )


@router.post(
    "/{collection_id}/documents",
    response_model=AddDocumentsResponse,
    summary="Add documents to a collection",
    description="""Add processed documents to a collection.
    
    This endpoint adds processed documents to a ChromaDB collection.
    
    Example:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/documents' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "documents": [
          {
            "text": "Document content here...",
            "metadata": {
              "source": "file.txt",
              "chunk_index": 0
            }
          }
        ]
      }'
    ```
    """,
    tags=["Ingestion"], # Add Ingestion tag here
    responses={
        200: {"description": "Documents added successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection not found"}
    }
)
async def add_documents(
    collection_id: int,
    request: AddDocumentsRequest,
    # token: str = Depends(verify_token), # Token verified by router dependency
    db: Session = Depends(get_db)
):
    """Add documents to a collection.
    
    Args:
        collection_id: ID of the collection
        request: Request with documents to add
        # token: Authentication token # Removed
        db: Database session
        
    Returns:
        Status information about the operation
        
    Raises:
        HTTPException: If collection not found or adding documents fails
    """
    try:
        # Ensure collection exists before adding documents (using helper)
        _get_and_validate_collection(db, collection_id)
        
        result = IngestionService.add_documents_to_collection(
            db=db,
            collection_id=collection_id,
            documents=request.documents
        )
        
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add documents to collection: {str(e)}"
        )


@router.post(
    "/{collection_id}/ingest-file",
    response_model=AddDocumentsResponse,
    summary="Ingest a file directly into a collection",
    description="""Upload, process, and add a file to a collection in one operation.
    
    This endpoint combines file upload, processing with an ingestion plugin, and adding 
    to the collection in a single operation.
    
    Example for simple_ingest plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/ingest-file' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -F 'file=@/path/to/document.txt' \
      -F 'plugin_name=simple_ingest' \
      -F 'plugin_params={\"chunk_size\": 1000, \"chunk_unit\": \"char\", \"chunk_overlap\": 200}' # Note JSON escaping
    ```
    
    Parameters for simple_ingest plugin:
    - chunk_size: Size of each chunk (default: 1000)
    - chunk_unit: Unit for chunking (char, word, line) (default: char)
    - chunk_overlap: Number of units to overlap between chunks (default: 200)
    """,
    tags=["Ingestion"], # Add Ingestion tag here
    responses={
        200: {"description": "File ingested successfully"},
        400: {"description": "Invalid plugin parameters"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or plugin not found"},
        500: {"description": "Error processing file or adding to collection"}
    }
)
async def ingest_file_to_collection(
    collection_id: int,
    file: UploadFile = File(...),
    plugin_name: str = Form(...),
    plugin_params: str = Form("{}"),
    # token: str = Depends(verify_token), # Token verified by router dependency
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = None # Add BackgroundTasks dependency
):
    """Ingest a file directly into a collection using a specified plugin.
    
    This endpoint combines file upload, processing with the specified plugin,
    and adding to the collection in a single operation.
    
    Args:
        collection_id: ID of the collection
        file: The file to upload and ingest
        plugin_name: Name of the ingestion plugin to use
        plugin_params: JSON string of parameters for the plugin
        # token: Authentication token # Removed
        db: Database session
        background_tasks: FastAPI background tasks
        
    Returns:
        Status information about the ingestion operation
        
    Raises:
        HTTPException: If collection not found, plugin not found, or ingestion fails
    """
    # Get and validate the collection using the helper
    collection, collection_name = _get_and_validate_collection(db, collection_id)

    # Check if plugin exists
    plugin = IngestionService.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(
            status_code=404,
            detail=f"Ingestion plugin '{plugin_name}' not found"
        )
    
    # Parse plugin parameters
    try:
        params = json.loads(plugin_params)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in plugin_params"
        )
    
    try:
        # Step 1: Upload file (this step remains synchronous)
        file_info = IngestionService.save_uploaded_file(
            file=file,
            owner=collection["owner"] if isinstance(collection, dict) else collection.owner,
            collection_name=collection_name
        )
        file_path = file_info["file_path"]
        file_url = file_info["file_url"]
        original_filename = file_info["original_filename"]
        owner = collection["owner"] if isinstance(collection, dict) else collection.owner
        
        # Step 2: Register the file in the FileRegistry with PROCESSING status
        file_registry = IngestionService.register_file(
            db=db,
            collection_id=collection_id,
            file_path=file_path,
            file_url=file_url,
            original_filename=original_filename,
            plugin_name=plugin_name,
            plugin_params=params,
            owner=owner,
            document_count=0,  # Will be updated after processing
            content_type=file.content_type,
            status=FileStatus.PROCESSING  # Set initial status to PROCESSING
        )
        
        # Step 3: Schedule background task for processing and adding documents
        def process_file_in_background(file_path: str, plugin_name: str, params: dict, 
                                     collection_id: int, file_registry_id: int):
            try:
                # Create a new session for the background task
                from database.connection import SessionLocal
                db_background = SessionLocal()
                
                try:
                    # Step 3.1: Process file with plugin
                    documents = IngestionService.ingest_file(
                        file_path=file_path,
                        plugin_name=plugin_name,
                        plugin_params=params
                    )
                    
                    # Step 3.2: Add documents to collection
                    result = IngestionService.add_documents_to_collection(
                        db=db_background,
                        collection_id=collection_id,
                        documents=documents
                    )
                    
                    # Step 3.3: Update file registry with completed status and document count
                    IngestionService.update_file_status(
                        db=db_background, 
                        file_id=file_registry_id, 
                        status=FileStatus.COMPLETED
                    )
                    
                    # Update document count
                    file_reg = db_background.query(FileRegistry).filter(FileRegistry.id == file_registry_id).first()
                    if file_reg:
                        file_reg.document_count = len(documents)
                        db_background.commit()
                    
                finally:
                    db_background.close()
                    
            except Exception as e:
                print(f"ERROR: [background_task] Failed to process file {file_path}: {str(e)}")

                # Update file status to FAILED
                try:
                    from database.connection import SessionLocal
                    db_error = SessionLocal()
                    try:
                        IngestionService.update_file_status(
                            db=db_error, 
                            file_id=file_registry_id, 
                            status=FileStatus.FAILED
                        )
                    finally:
                        db_error.close()
                except Exception:
                    print(f"ERROR: [background_task] Could not update file status to FAILED")
        
        # Add the task to background tasks
        background_tasks.add_task(
            process_file_in_background, 
            file_path, 
            plugin_name, 
            params, 
            collection_id, 
            file_registry.id
        )
        
        # Return immediate response with file information
        return {
            "collection_id": collection_id,
            "collection_name": collection_name,
            "documents_added": 0,  # Initially 0 since processing will happen in background
            "success": True,
            "file_path": file_path,
            "file_url": file_url,
            "original_filename": original_filename,
            "plugin_name": plugin_name,
            "file_registry_id": file_registry.id,
            "status": "processing"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}"
        )


# Query endpoint nested under collections

@router.post(
    "/{collection_id}/query",
    response_model=QueryResponse,
    summary="Query a collection",
    description="""Query a collection using a specified plugin.
    
    This endpoint performs a query on a collection using the specified query plugin.
    
    Example for simple_query plugin:
    ```bash
    curl -X POST 'http://localhost:9090/collections/1/query' \
      -H 'Authorization: Bearer 0p3n-w3bu!' \
      -H 'Content-Type: application/json' \
      -d '{
        "query_text": "What is the capital of France?",
        "top_k": 5,
        "threshold": 0.5,
        "plugin_params": {}
      }'
    ```
    
    Parameters for simple_query plugin:
    - query_text: The text to query for
    - top_k: Number of results to return (default: 5)
    - threshold: Minimum similarity threshold (0-1) (default: 0.0)
    """,
    tags=["Query"], # Add Query tag here
    responses={
        200: {"description": "Query results"},
        400: {"description": "Invalid query parameters"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection or query plugin not found"}
    }
)
async def query_collection(
    collection_id: int,
    request: QueryRequest,
    plugin_name: str = Query("simple_query", description="Name of the query plugin to use"),
    # token: str = Depends(verify_token), # Token verified by router dependency
    db: Session = Depends(get_db)
):
    """Query a collection using a specified plugin.
    
    Args:
        collection_id: ID of the collection to query
        request: Query request parameters
        plugin_name: Name of the query plugin to use
        # token: Authentication token # Removed
        db: Database session
        
    Returns:
        Query results
        
    Raises:
        HTTPException: If collection not found, plugin not found, or query fails
    """
    # Get and validate the collection using the helper
    collection, collection_name = _get_and_validate_collection(db, collection_id)

    # Prepare plugin parameters
    plugin_params = request.plugin_params or {}
    
    # Add standard parameters if not in plugin_params
    if "top_k" not in plugin_params and request.top_k is not None:
        plugin_params["top_k"] = request.top_k
    if "threshold" not in plugin_params and request.threshold is not None:
        plugin_params["threshold"] = request.threshold
    
    try:
        # Query the collection
        result = QueryService.query_collection(
            db=db,
            collection_id=collection_id,
            query_text=request.query_text,
            plugin_name=plugin_name,
            plugin_params=plugin_params
        )
        
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query collection: {str(e)}"
        )


# File Registry Endpoints related to Collections

@router.get(
    "/{collection_id}/files",
    response_model=List[FileRegistryResponse],
    summary="List files in a collection",
    description="Get a list of all files in a collection",
    tags=["Files"], # Add Files tag
    responses={
        200: {"description": "List of files in the collection"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "Collection not found"},
        500: {"description": "Server error"}
    }
)
async def list_files(
    collection_id: int,
    # token: str = Depends(verify_token), # Already handled by router dependency
    db: Session = Depends(get_db),
    status: str = Query(None, description="Filter by status (completed, processing, failed, deleted)")
):
    """List all files in a collection.
    
    Args:
        collection_id: ID of the collection
        db: Database session
        status: Optional filter by status
        
    Returns:
        List of file registry entries
        
    Raises:
        HTTPException: If collection not found
    """
    # Use CollectionsService which is already imported
    return CollectionsService.list_files(collection_id, db, status)


# Note: The endpoint below does NOT use the /collections prefix from the router
@router.put(
    "/files/{file_id}/status", # Define path relative to root
    response_model=FileRegistryResponse,
    summary="Update file status",
    description="Update the status of a file in the registry",
    tags=["Files"], # Add Files tag
    responses={
        200: {"description": "File status updated successfully"},
        401: {"description": "Unauthorized - Invalid or missing authentication token"},
        404: {"description": "File not found"},
        500: {"description": "Server error"}
    }
)
async def update_file_status(
    file_id: int,
    status: str = Query(..., description="New status (completed, processing, failed, deleted)"),
    # token: str = Depends(verify_token), # Already handled by router dependency
    db: Session = Depends(get_db)
):
    """Update the status of a file in the registry.
    
    Args:
        file_id: ID of the file registry entry
        status: New status
        db: Database session
        
    Returns:
        Updated file registry entry
        
    Raises:
        HTTPException: If file not found or status invalid
    """
    # Use CollectionsService which is already imported
    return CollectionsService.update_file_status(file_id, status, db) 