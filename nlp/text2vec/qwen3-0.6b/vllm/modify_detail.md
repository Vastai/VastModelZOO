## Evalscope修改细节

### 以下1-6修改，可通过git-patch一次更新

- 参考：[evalscope.patch](./evalscope.patch)

    ```bash
    git clone https://github.com/modelscope/evalscope.git
    cd evalscope
    git checkout v1.3.0
    git apply evalscope.patch
    ```

- 使用源码安装

    ```bash
    cd evalscope
    #安装依赖
    pip install -e .
    #安装RAGEval评测后端
    pip install '.[rag]'
    ```
### modify.1 修改one_stage_eval函数实现
- **修改**[evalscope/evalscope/backend/rag_eval/cmteb/task_template.py#L39](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/cmteb/task_template.py#L39)的one_stage_eval函数，修改内容如下：
    - 新增cross_encoder模型的运行实现
    - 增加save_predictions=True参数，实现把评估结果保存到指定目录
    ```python
    def one_stage_eval(
        model_args,
        eval_args,
    ) -> None:
        # load model
        model = EmbeddingModel.load(**model_args)
        print("model=", model)
        custom_dataset_path = eval_args.pop('dataset_path', None)
        # load task first to update instructions
        tasks = cmteb.TaskBase.get_tasks(task_names=eval_args['tasks'], dataset_path=custom_dataset_path)
        
        if model_args.get('is_cross_encoder', False):
            previous_results = model_args['model_kwargs']['previous_results']
            for i in range(len(tasks)):
                print(f'Running evaluation for {tasks[i]}...')
                print(f'Previous results: {previous_results[i]}')
                evaluation = mteb.MTEB(tasks=[tasks[i]])
                results = evaluation.run(
                    model,
                    top_k=eval_args['top_k'],
                    save_predictions=True,
                    output_folder=eval_args['output_folder'],
                    previous_results=previous_results[i],
                    overwrite_results=True,
                    hub=eval_args['hub'],
                    limits=eval_args['limits'],
                    encode_kwargs=model_args.get('encode_kwargs', {}),
                )
                model.task_names.pop(0)

        else:
            evaluation = mteb.MTEB(tasks=tasks)
            eval_args['encode_kwargs'] = model_args.get('encode_kwargs', {})
            
            # run evaluation
            results = evaluation.run(model, save_predictions=True, **eval_args)

        # save and log results
        show_results(eval_args['output_folder'], model, results)
    ```

### modify.2 修改STS任务的QBQTC函数实现
- **修改**[evalscope/evalscope/backend/rag_eval/cmteb/tasks/STS.py#L297](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/cmteb/tasks/STS.py#L297)的QBQTC函数，修改内容如下：
    - 新增metadata_dict属性，用于保存QBQTC任务的min_score和max_score
    ```python
    class QBQTC(AbsTaskSTS):
        metadata = TaskMetadata(
            name='QBQTC',
            dataset={
                'path': 'C-MTEB/QBQTC',
                'revision': '790b0510dc52b1553e8c49f3d2afb48c0e5c48b7',
            },
            description='',
            reference='https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset',
            type='STS',
            category='s2s',
            modalities=['text'],
            eval_splits=['test'],
            eval_langs=['cmn-Hans'],
            main_score='cosine_spearman',
            date=None,
            domains=None,
            task_subtypes=None,
            license=None,
            annotations_creators=None,
            dialect=None,
            sample_creation=None,
            bibtex_citation=None,
            descriptive_stats={
                'n_samples': None,
                'avg_character_length': None
            },
        )
        @property
        def metadata_dict(self) -> dict[str, str]:
            metadata_dict = super().metadata_dict
            metadata_dict['min_score'] = 0
            metadata_dict['max_score'] = 1
            return metadata_dict
    ```
### modify.3 修改BaseModel类实现
- **修改**[evalscope/backend/rag_eval/utils/embedding.py#L21](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/utils/embedding.py#L21)的BaseModel类，修改内容如下：
    - 新增instruction_template属性，用于指定instruction_template
    - 新增instruction_dict属性，用于保存instruction_dict
    - 新增get_instruction函数，用于获取instruction
    - 新增format_instruction函数，用于格式化instruction

    ```python
    class BaseModel(Embeddings):

    def __init__(
        self,
        model_name_or_path: str = '',
        max_seq_length: int = 512,
        prompt: Optional[str] = None,
        prompts: Optional[Dict[str, str]] = None,
        revision: Optional[str] = 'master',
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.model_kwargs = kwargs.pop('model_kwargs', {})

        self.config_kwargs = kwargs.pop('config_kwargs', {})
        self.config_kwargs['trust_remote_code'] = True

        self.encode_kwargs = kwargs.pop('encode_kwargs', {})
        self.encode_kwargs['convert_to_tensor'] = True

        self.prompt = prompt
        self.prompts = prompts if prompts else {}
        self.revision = revision
        self.framework = ['PyTorch']
        self.instruction_dict = dict()
        if self.model_kwargs['instruction_dict_path'] is not None:
            instruction_dict_path =  self.model_kwargs['instruction_dict_path']
            with open(instruction_dict_path) as f:
                    self.instruction_dict = json.load(f)
        self.instruction_template = self.model_kwargs.get('instruction_template', None)

    def get_instruction(self, task_name, prompt_type):
        sym_task = False
        instruction = None
        print("task_name=", task_name)
        if task_name in self.instruction_dict:
            instruction = self.instruction_dict[task_name]
            print("instruction from dict=", instruction)
            if isinstance(instruction, dict):
                instruction = instruction.get(prompt_type, "")
                print("instruction =", instruction)
                sym_task = True
        task_type = mteb.get_tasks(tasks=[task_name])[0].metadata.type
        if 'Retrieval' in task_type and not sym_task and prompt_type != 'query':
            return ""
        if task_type in ["STS", "PairClassification"]:
            return "Retrieve semantically similar text"
        if task_type in "Bitext Mining":
            return "Retrieve parallel sentences"
        if 'Retrieval' in task_type and prompt_type == 'query' and instruction is None:
            instruction = "Retrieval relevant passage for the given query."
        return instruction
        
    def format_instruction(self, instruction, prompt_type):
        if instruction is not None and len(instruction.strip()) > 0:
            instruction = self.instruction_template.format(instruction)
            return instruction
        return ""
    ```
### modify.4 修改CrossEncoderModel类的实现
- **修改**[evalscope/backend/rag_eval/utils/embedding.py#L154](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/utils/embedding.py#L154)的CrossEncoderModel类，修改内容如下：

    - 修改初始化函数
    - 新增send_post_request函数，用于发送post请求
    - 修改predict函数，用于调用api进行预测

    ```python
    class CrossEncoderModel(BaseModel):

    # def __init__(self, model_name_or_path: str, **kwargs):
    #     super().__init__(model_name_or_path, **kwargs)
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name')
        super().__init__(model_name_or_path=self.model_name, **kwargs)

        self.framework = ['Sentence Transformers', 'PyTorch']

        # self.model = CrossEncoder(
        #     self.model_name_or_path,
        #     trust_remote_code=True,
        #     max_length=self.max_seq_length,
        #     automodel_args=self.model_kwargs,
        # )
        # self.tokenizer = self.model.tokenizer
        # # set pad token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # if ('pad_token_id' not in self.model.config) or (self.model.config.pad_token_id is None):
        #     self.model.config.update({'pad_token_id': self.tokenizer.eos_token_id})

        # self.supported_encode_params = get_supported_params(self.model.predict)
        self.url = kwargs.get('api_base')
        self.task_names = kwargs["model_kwargs"].get('task_name')
        print("task_names: ", self.task_names)
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

    def send_post_request(self, url, payload, headers=None):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def predict(self, sentences: List[List[str]], **kwargs) -> Tensor:
        # for key in list(kwargs.keys()):
        #     if key not in self.supported_encode_params:
        #         kwargs.pop(key)
        # self.encode_kwargs.update(kwargs)

        # if len(sentences[0]) == 2:  # Note: For mteb retrieval task
        #     processed_sentences = []
        #     for query, docs in sentences:
        #         processed_sentences.append((self.prompt + query, docs))
        #     sentences = processed_sentences
        # embeddings = self.model.predict(sentences, **self.encode_kwargs)
        # assert isinstance(embeddings, Tensor)
        # return embeddings
        relevance_scores = []
        print("sentences: ",sentences)
        task_name = self.task_names[0]
        instruction = self.get_instruction(task_name, 'query')
        print('instruction===', instruction)
        for item in sentences:
            query = self.query_template.format(prefix=self.prefix, instruction=instruction, query=item[0])
            docs = self.document_template.format(doc=item[1], suffix=self.suffix)
            
            payload = {"model": self.model_name, "query": query, "documents": [docs]}

            response = self.send_post_request(self.url, payload)
            if 'results' in response:
                for item in response['results']:
                    relevance_scores.append(item['relevance_score'])
            else:
                print("response: ",response)
        print("relevance_scores: ",relevance_scores)
        return relevance_scores
    ```

### modify.5 修改APIEmbeddingModel类的encode函数实现
- **修改**[evalscope/backend/rag_eval/utils/embedding.py#L192](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/utils/embedding.py#L192)的APIEmbeddingModel类的encode函数，修改内容如下：

    - 利用get_instruction函数，获取instruction
    - 利用format_instruction函数，格式化instruction
    - 在encode函数中，将instruction添加到texts中，去除prompt参数
    
    ```python
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Tensor:
        # pop unused kwargs
        extra_params = {}
        for key in list(kwargs.keys()):
            if key not in self.supported_encode_params:
                extra_params[key] = kwargs.pop(key)
        self.encode_kwargs.update(kwargs)

        # set prompt if provided
        prompt = None
        prompt_type = extra_params.pop('prompt_type', '')
        task_name = extra_params.pop('task_name', '')
        # if prompt_type and prompt_type == PromptType.query:
        #     prompt = self.get_prompt(task_name)

        print('prompt_type===', prompt_type)
        instruction = self.get_instruction(task_name, prompt_type)
        if self.instruction_template:
            instruction = self.format_instruction(instruction, prompt_type)
        logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")

        if isinstance(texts, str):
            texts = [texts]

        embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            # set prompt if provided
            # if prompt is not None:
            #     batch_texts = [prompt + text for text in texts[i:i + self.batch_size]]
            # else:
            #     batch_texts = texts[i:i + self.batch_size]
            if instruction is not None:
                batch_texts = [instruction + text for text in texts[i:i + self.batch_size]]
            else:
                batch_texts = texts[i:i + self.batch_size]
            response = self.model.embed_documents(batch_texts, chunk_size=self.batch_size)
            embeddings.extend(response)
        return torch.tensor(embeddings)
    ```
### modify.6 修改EmbeddingModel类的load函数实现
- **修改**[evalscope/backend/rag_eval/utils/embedding.py#L246](https://github.com/modelscope/evalscope/blob/v1.3.0/evalscope/backend/rag_eval/utils/embedding.py#L246)的EmbeddingModel类的load函数，修改内容如下：

    - 增加is_cross_encoder==False的判断，如果model_name是提供的且is_cross_encoder==False，则使用OpenAIEmbeddings
    - 修改CrossEncoderModel类的初始化参数
    
    ```python
    class EmbeddingModel:
    """Custom embeddings"""

    @staticmethod
    def load(
        model_name_or_path: str = '',
        is_cross_encoder: bool = False,
        hub: str = HubType.MODELSCOPE,
        revision: Optional[str] = 'master',
        **kwargs,
    ):
        if kwargs.get('model_name') and is_cross_encoder==False:
            # If model_name is provided, use OpenAIEmbeddings
            return APIEmbeddingModel(**kwargs)

        # If model path does not exist and hub is 'modelscope', download the model
        if not os.path.exists(model_name_or_path) and hub == HubType.MODELSCOPE:
            model_name_or_path = download_model(model_name_or_path, revision)

        # Return different model instances based on whether it is a cross-encoder and pooling mode
        if is_cross_encoder:
            return CrossEncoderModel(
                **kwargs
            )
            # return CrossEncoderModel(
            #     model_name_or_path,
            #     revision=revision,
            #     **kwargs,
            # )
        else:
            return SentenceTransformerModel(
                model_name_or_path,
                revision=revision,
                **kwargs,
            )
    ```