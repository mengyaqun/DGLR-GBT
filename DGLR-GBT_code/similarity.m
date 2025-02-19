function [sim_mat] = similarity(x, sim_measure)
% Compute similarity between rows of the matrix X.X - we need to compute similarity between rows of this matrix
% sim_measure - e.g. cosine
n_rows = size(x,1);
sim_mat = zeros(n_rows,n_rows);
for row1=1:n_rows
    x_row1 = x(row1,:);
    for row2=1:n_rows
        % Computing similarity between X_row1 and X_row2. Similarity is the measure of weight of the edge between X_row1 and X_row2 vertices
        x_row2 = x(row2,:);
        % Indices of the values which are present in both X_row1 and X_row2
        co_indices = x_row1.*x_row2 > 0;
        % Cosine similarity
        if size(x_row1(co_indices), 2) > 0
            cos_sim = 1 - pdist([x_row1(co_indices);x_row2(co_indices)], sim_measure);
        else
            cos_sim = 0;
        end
        sim_mat(row1,row2) = cos_sim;
    end
end

end
