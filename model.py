class AttributeMapper():

  def common(self,a,b): 
    c = [value for value in a if value in b] 
    return c
  
  def ispresent(self,a,b):
    count=0
    for i in a:
      if(i in b):
        count+=1
    return count

  def __init__(self, path, attribute_similarity_ratio, max_depth):
    self.mapped_attributes = {}
    self.mapped_subentities = {}
    self.entity_list = []
    self.path = path
    self.attribute_ratio = attribute_similarity_ratio
    self.max_depth = max_depth
    self.id_counter = 0
    self.name_id = {}
    self.edges_index = []
    self.attribute_vectors = []
    self.ent_class = []
    self.max_attr_len = -1
    self.tot_len = 0
    import spacy, nltk
    self.coref_nlp = spacy.load("en_coreference_web_trf")
    # nltk.download("averaged_perceptron_tagger")

  def set_params(self, path, attribute_similarity_ratio, max_depth):
    self.clear_all()
    self.mapped_attributes = {}
    self.mapped_subentities = {}
    self.entity_list = []
    self.path = path
    self.attribute_ratio = attribute_similarity_ratio
    self.max_depth = max_depth
    self.id_counter = 0
    self.name_id = {}
    self.edges_index = []
    self.attribute_vectors = []
    self.max_attr_len = -1
    self.tot_len = 0
    self.ent_class = []


  def clear_all(self):
    self.text = ''
    self.pronoun_clusters = None
    self.mapped_attributes = {}
    self.mapped_subentities = {}
    self.entity_list = []
    self.path = ''
    self.attribute_ratio = 0
    self.max_depth = 0
    self.id_counter = 0
    self.name_id = {}
    self.edges_index = []
    self.attribute_vectors = []
    self.max_attr_len = -1
    self.tot_len = 0
    self.ent_class = []
  
  def extract(self):
    import re
    import fitz
    text=""
    doc = fitz.open(self.path)
    i=0
    for i in range(doc.page_count):
      page = doc[i]
      words = page.get_text("words")
      for i in words:
        text+=(i[4]+" ")
    text = re.sub('[^a-zA-Z0-9.]', ' ', text)
    text = re.sub('\s\s+', ' ', text)
    print("Text extraction and cleaning completed!!")
    self.text = text
  
  def pronoun_mapping(self):
    coref_doc = self.coref_nlp(self.text)
    self.pronoun_clusters = coref_doc.spans.values() # Pronoun clusters 
    self.coref_doc = coref_doc
    print("Pronoun mapping and clustering completed!!")
    return self.pronoun_clusters

  def extract_attributes(self):
    import nltk
    self.extract()
    self.pronoun_mapping()
    coref_doc = self.coref_doc
    pronoun_clusters = self.pronoun_clusters
    descriptive_criterion = ['RB', 'JJ', 'JJR', 'JJS', 'CD']
    noun_criterion = ['NN', 'NNS', 'NNP']
    criterion = descriptive_criterion + noun_criterion
    entity_dict = {}
    for cluster in pronoun_clusters:
      temp_dict = {}
      all_attri = []
      entity_name = ''
      tagged_entity = nltk.tag.pos_tag(cluster[0].text.split())
      k = 0
      num = False
      entity_name_list = []
      for word,tag in tagged_entity:
        if num==True:
          num = False
        elif tag in descriptive_criterion:
          all_attri.append(word.lower())
        elif tag in noun_criterion:
          all_attri.append(word.lower())
          entity_name_list.append(word.lower())
        k = k + 1
      entity_name = " ".join(entity_name_list)
      if(entity_name == ""): continue
      temp_dict['entity'] = entity_name
      for i in range(1,len(cluster)):
        num = False
        sentence = coref_doc[cluster[i].end:].text.split(". ")[0]
        tagged_sentence = nltk.tag.pos_tag(sentence.split())
        k = 0
        for word, tag in tagged_sentence:
          if tag in criterion and word not in all_attri:
            all_attri.append(word.lower())
          k = k + 1
      temp_dict['attributes'] = all_attri
      if(len(all_attri)==0): continue
      self.max_attr_len = max(self.max_attr_len, len(all_attri))
      try:
        for attri in all_attri:
          if attri not in entity_dict[self.name_id[entity_name.lower()]]['attributes']:
            entity_dict[self.name_id[entity_name.lower()]]['attributes'].append(attri)
      except:
        self.name_id[entity_name.lower()] = self.id_counter
        self.id_counter = self.id_counter + 1
        entity_dict[self.name_id[entity_name.lower()]] = temp_dict
    self.mapped_attributes = entity_dict
    print("Attributes mapped!!")
    self.tot_len = len(entity_dict)
    self.ent_class = [0]*self.tot_len
    return entity_dict
  
  def extract_subentities(self, current_graph, current_depth=1):
    if(current_graph=={} or current_depth>self.max_depth):
      return 
    from copy import deepcopy
    visited = [0]*self.tot_len
    deletion_list = []
    k=1
    for i, ent in current_graph.items():
      temp = {}
      if(visited[i]==1):
        k = k + 1
        continue
      visited[i] = 1
      for j, match_ent in list(current_graph.items())[k:]:
        if(self.ispresent(ent['attributes'], match_ent['attributes'])>=max(len(ent['attributes'])*self.attribute_ratio, 1) or self.ispresent(ent['entity'].split(), match_ent['entity'].split())>=max(len(ent['entity'].split())*self.attribute_ratio, 1)):
          visited[j] = 1
          temp[j] = deepcopy(match_ent)
          deletion_list.append(j)
      k = k + 1
      if(temp!={}):
        ent['sub_entity'] = temp
    if(current_depth+1>self.max_depth):
      return current_graph
    for delete in deletion_list:
      try:
        del current_graph[delete]
      except:
        pass
    for id, ent in current_graph.items():
      try:
        ent['sub_entity'] = self.extract_subentities(deepcopy(ent['sub_entity']), current_depth+1)
      except:
        pass
    if(current_depth==1):
      self.subentities = current_graph
      for id, ent in current_graph.items():
        self.ent_class[id] = 1
    return current_graph
  
  def getedgesrecursive(self, subent, id_ent):
    for id_sub, sub in subent.items():
      self.edges_index.append([id_ent, id_sub])
      try:
        self.getedgesrecursive(sub['sub_entity'], id_sub)
      except:
        pass

  def get_tensors(self):
    subentities = self.subentities
    for id, entity in subentities.items():
      try:
        self.getedgesrecursive(entity['sub_entity'], id)
      except:
        pass
    return self.edges_index
  
  def get_graph(self):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for id, ent in self.mapped_attributes.items():
      if(self.ent_class[id]==1):
        G.add_node(ent['entity'], color='red',style='filled')
      else:
        G.add_node(ent['entity'], color='green',style='filled')
    for id1, id2 in self.edges_index:
      G.add_edge(self.mapped_attributes[id1]['entity'], self.mapped_attributes[id2]['entity'])
    colored_dict = nx.get_node_attributes(G, 'color')
    default_color = 'yellow'
    color_seq = [colored_dict.get(node, default_color) for node in G.nodes()]
    plt.figure(figsize=(20,15))
    nx.draw(G, with_labels=True, node_color=color_seq, edge_cmap=plt.cm.Blues)
    plt.savefig(f'static/images/{self.path.split("/")[-1].split(".")[0]}.png')
    plt.show()

mapper = AttributeMapper("", 0, 0)

import pickle

pickle.dump(mapper, open("extractattri.pkl", "wb"))