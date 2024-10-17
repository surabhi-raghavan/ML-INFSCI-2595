import pandas as pd
import numpy as np 
from pandas.api.types import is_numeric_dtype

class PrismAlgorithm:
    def __init__(self, minc=0, min_prob=0, nbins=0):

        self.minc = minc
        self.min_prob = min_prob
        self.nbins = nbins
        self.bin_ranges = {}
        self.default_target = None
        self.pred_dictionary = {}
        self.target_column = ""
        self.int_to_values_map = {}
        
    def getNewTerm(self, df, target_col, target_val, exclude_column):
        max_prob = -1
        max_prob_freq = 0
        term = None
        for col_name in df.drop(columns=[target_col]).columns:
            if col_name in exclude_column:
                continue
            for val in df[col_name].unique():
                sub_df_1 = df[df[col_name] == val]
                sub_df_2 = df[(df[col_name] == val) & (df[target_col] == target_val)]
                prob = len(sub_df_2) / len(sub_df_1)
                freq = len(sub_df_2)
                if (prob > max_prob) or ((prob == max_prob) and (freq > max_prob_freq)):
                    max_prob = prob
                    max_prob_freq = freq
                    term = (col_name, val, prob, freq)
        return term

    def getNewRule(self, df_rule, target_col, target_val):
        term = []
        exclude_column = []  
        df = df_rule.copy()
        target_val_remaining = df[target_col].tolist().count(target_val)
        len_df = len(df)
        
        while target_val_remaining < len_df:
            term = self.getNewTerm(df, target_col, target_val, exclude_column)
            if term is None:
                break
            col, val, prob, freq = term
            exclude_column.append(col)
            if len(exclude_column) == (len(df.columns) - 1):
                break
            df = df[(df[col] == val)]
            term.append((term, df))
            target_val_remaining = df[target_col].tolist().count(target_val)
            len_df = len(df)

        max_prob = 0
        max_prob_idx = -1
        max_prob_df = None
        for term_idx, term in enumerate(term):
            if term[0][2] > max_prob:
                max_prob = term[0][2]
                max_prob_idx = term_idx
                max_prob_df = term[1]
        if max_prob_idx == -1:
            return None, None
        optimal_term = term[:max_prob_idx + 1]
        optimal_term = [x for x, y in optimal_term]
        formatted_terms = []
        for t in optimal_term:
            if t[1] == "NONE":
                formatted_terms.append(t)
            else:
                formatted_terms.append((t[0], self.int_to_values_map[t[0]][t[1]], t[2], t[3]))

        return formatted_terms, max_prob_df

    def getRule(self, df_full, target_col, target_val):
       
        rules_for_val = []
        num_rows_curr_target = df_full[target_col].tolist().count(target_val)
        df = df_full.copy() 
        is_first_rule = True
        target_val_remaining = df[target_col].tolist().count(target_val)
        while target_val_remaining > 0:

            num_rows_curr_target_remaining = df[target_col].tolist().count(target_val)
            term, rule_df = self.getNewRule(df, target_col, target_val)
            if not term:
                remaining_indices = df[df[target_col] == target_val].index
                rule_df = df.loc[remaining_indices]
                term = [('Default', 'Any', 1.0, len(rule_df))]
                df = df.drop(index=remaining_indices)
                target_val_remaining = 0
            else:
                df = df.loc[list(set(df.index) - set(rule_df.index))]
                target_val_remaining = df[target_col].tolist().count(target_val)

            if target_val in self.pred_dictionary:
                self.pred_dictionary[target_val].append(term)
            else:
                self.pred_dictionary[target_val] = [term]

            rule_str = ""
            for term in term:
                if term[0] == 'Default':
                    rule_str += "Default rule (covers remaining instances)"
                else:
                    rule_str += str(term[0]) + " = " + str(term[1]) + " AND "
            if 'AND' in rule_str:
                rule_str = rule_str[:-5]  # Remove the trailing "AND"
            num_matching_target = rule_df[target_col].tolist().count(target_val)
            rule_str += "\n   Support:  "
            if is_first_rule:
                rule_str += f"the target has value: '{self.int_to_values_map[target_col][target_val]}' for "
                rule_str += f"{term[-1][2] * 100:.3f}% of the {num_matching_target} rows matching the rule "
                rule_str += "\n   Coverage: "
                rule_str += f"the rule matches: {num_matching_target} out of {num_rows_curr_target} rows "
                rule_str += f"for target value: '{self.int_to_values_map[target_col][target_val]}'. This is:"
            else:
                rule_str += f"The target has value: '{self.int_to_values_map[target_col][target_val]}' for "
                rule_str += f"{term[-1][2] * 100:.3f}% of the {num_matching_target} remaining rows matching the "
                rule_str += "rule"
                rule_str += "\n   Coverage: "
                rule_str += f"The rule matches: {num_matching_target} out of {num_rows_curr_target_remaining} rows "
                rule_str += f"remaining for target value: '{self.int_to_values_map[target_col][target_val]}'. This is:"
                rule_str += f"\n      {(num_matching_target * 100.0 / num_rows_curr_target_remaining):.3f}% of "
                rule_str += f"remaining rows for target value: '{self.int_to_values_map[target_col][target_val]}'"
            rule_str += f"\n      {(num_matching_target * 100.0 / num_rows_curr_target):.3f}% of total rows for target "
            rule_str += f"value: '{self.int_to_values_map[target_col][target_val]}'"
            rule_str += f"\n      {(num_matching_target * 100.0 / len(df_full)):.3f}% of total rows in data"
            rules_for_val.append(rule_str)
            is_first_rule = False

        return rules_for_val
    def display(self, df, target_col, rules_dict, display_stats):
        for target_val in rules_dict:
            print()
            print('........................................................................')
            print(f"Target: {self.int_to_values_map[target_col][target_val]}")
            print('........................................................................')
            if len(rules_dict[target_val]) == 0 and display_stats:
                print((f"  No rules imputed for target value {self.int_to_values_map[target_col][target_val]}. There "
                       f"are {df[target_col].tolist().count(target_val)} rows for this class."))
            for r in rules_dict[target_val]:
                if display_stats: 
                    print(r)
                else:
                    print(r.split("\n")[0])

    def get_prism_rules(self, df, target_col, display_stats=True):
        new_vals_dict = {}
        for col_name in df.columns:
            if is_numeric_dtype(df[col_name]) and (df[col_name].nunique() > 10):
                try:
                    vals = [f"Bin_{x}" for x in range(self.nbins)]
                    bin_ids, bin_ranges = pd.qcut(df[col_name], self.nbins, labels=vals, retbins=True)
                    self.bin_ranges[col_name] = bin_ranges
                    vals_to_int_map = {x: y for x, y in zip(vals, range(len(vals)))}
                    int_to_vals_map = {y: x for x, y in zip(vals, range(len(vals)))}
                    new_vals_dict[col_name] = bin_ids.map(vals_to_int_map)
                    self.int_to_values_map[col_name] = int_to_vals_map
                except:
                    pass
            else:
                vals = df[col_name].unique()
                vals_to_int_map = {x: y for x, y in zip(vals, range(len(vals)))}
                int_to_vals_map = {y: x for x, y in zip(vals, range(len(vals)))}
                new_vals_dict[col_name] = df[col_name].map(vals_to_int_map)
                self.int_to_values_map[col_name] = int_to_vals_map
        df = pd.DataFrame(new_vals_dict)
        target_vals = sorted(df[target_col].unique())

        rules_dict = {}
        for target_val in target_vals:

            rules_dict[target_val] = self.getRule(df, target_col, target_val)

        self.display(df, target_col, rules_dict, display_stats)

        self.default_target = df[target_col].mode().values[0]
        self.target_column = target_col

        return rules_dict

    def predict(self, X_in, leave_unknown=False):
        X = X_in.copy()
        X = X.reset_index(drop=True)
        if leave_unknown:
            ret = ["NO PREDICTION"] * len(X)
        else:
            ret = [self.default_target] * len(X)
        is_set = [False] * len(X)

        for col_name in X.columns:
            if col_name not in self.bin_ranges:
                continue
            bin_ranges = self.bin_ranges[col_name]
            for i in range(len(X)):
                v = X.loc[i, col_name]
                for bin_idx, bin_limit in enumerate(bin_ranges):
                    if v < bin_limit:
                        X.loc[i, col_name] = bin_idx - 1
                        break
            if col_name in self.int_to_values_map:
                X[col_name] = X[col_name].map(self.int_to_values_map[col_name])

        for i in range(len(X)):
            row = X.iloc[i]
            found_rule = False
            for key in self.pred_dictionary.keys():
                rules = self.pred_dictionary[key]
                for rule in rules:
                    all_terms_true = True
                    for term in rule:
                        term_feature_name = term[0]
                        term_value = term[1]
                        if term_feature_name == 'Default':
                            continue
                        if row[term_feature_name] != term_value:
                            all_terms_true = False
                            break
                    if all_terms_true:
                        ret[i] = key
                        is_set[i] = True
                        found_rule = True
                        break
                if found_rule:
                    break
        ret = pd.Series(ret).map(self.int_to_values_map[self.target_column])
        if leave_unknown:
            ret = ret.fillna("NO PREDICTION")
            ret = ret.astype(str)
        return ret

    def binRange(self):
        if self.bin_ranges is None:
            print("No columns were binned.")
        else:
            return self.bin_ranges


df1 = pd.read_csv("HW1/Training200.csv")
df1.head()

def create_class_label(df, is_training=True):
    df = df1.copy()
    df['Next Day Opening'] = df['Open'].shift(-1)
    if is_training:
        
        df.loc[df.index[-1], 'Next Day Opening'] = df.loc[df.index[-1], 'Close'] + 1 
    else:
        df = df[:-1]  
    df['Class'] = np.where(df['Next Day Opening'] > df['Close'], 'YES', 'NO')
    df = df.drop(columns=['Next Day Opening', 'Unnamed: 0', "Stock Splits"])
    return df

training_data = create_class_label(df1, is_training=True)
validation_data = create_class_label("HW1/Validation65.csv", is_training=False)
print(training_data.columns)

prism = PrismAlgorithm(nbins=10)
rules = prism.get_prism_rules(training_data, 'Class')
print(training_data.columns)

predictions = prism.predict(training_data.drop(columns=['Class']))

from sklearn.metrics import accuracy_score

X_val = validation_data.drop(columns=['Class'])
y_val = validation_data['Class']
y_pred = prism.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        