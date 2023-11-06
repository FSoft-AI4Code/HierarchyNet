from inputters import UNK_TYPE

class NodeTokenizer:
    def __init__(self, node_type_path, tokenizer):
        self.node_type_path = node_type_path
        node_types = list(map(lambda x: x.strip(), open(node_type_path).readlines()))
        self.node_type_mapping = dict(zip(node_types, range(len(node_types))))
        self.tokenizer = tokenizer
    def get_node_type_id(self, node_type):
        return self.node_type_mapping.get(node_type, self.node_type_mapping[UNK_TYPE])
    def get_node_token_ids(self, node_token):
        token_ids = self.tokenizer.encode(node_token).ids
        return token_ids
    def types_to_ids(self, node_types):
        return list(map(self.get_node_type_id, node_types))
    def tokens_to_ids(self, node_tokens):
        return list(map(self.tokenizer.token_to_id, node_tokens))