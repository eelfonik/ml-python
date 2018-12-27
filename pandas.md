- avoid automatic added first row as index when reading csv files: `data = pd.read_csv('path/to/file.csv', index_col=0)`

- DataFrame: sum up some columns and put the result in another column: 
`df['some col name'] = df.iloc[:, 1:10].sum(axis=1)`
https://stackoverflow.com/a/48841156

- tranpose a dataframe with `T`, see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.transpose.html

- For some european numeric format, ex: has `,` as thousands sperator, you need to apply the function: `df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',',''), errors='coerce'))
`, to convert the string with `,` to float

- pick only the rows with uppercase letter value in a column: `df[df['column_name'].str.match('^.*[A-Z]$')]`, see https://stackoverflow.com/a/49641518
