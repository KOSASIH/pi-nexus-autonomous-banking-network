class InputValidator
  def self.validate(input, rules)
    errors = {}
    rules.each do |field, rule|
      value = input[field]
      if value.nil? || value.empty?
        errors[field] = "cannot be blank"
      elsif rule[:type] == :string
        if value.length < rule[:min_length] || value.length > rule[:max_length]
          errors[field] = "must be between #{rule[:min_length]} and #{rule[:max_length]} characters"
        end
      elsif rule[:type] == :email
        if !value.match?(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/)
          errors[field] = "must be a valid email address"
        end
      elsif rule[:type] == :password
        if value.length < rule[:min_length]
          errors[field] = "must be at least #{rule[:min_length]} characters"
        end
      end
    end
    errors
  end
end
