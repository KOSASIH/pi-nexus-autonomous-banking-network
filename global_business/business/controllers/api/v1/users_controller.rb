class Api::V1::UsersController < ApplicationController
  before_action :authenticate_user!

  def show
    render json: current_user, status: :ok
  end

  def update
    if current_user.update(user_params)
      render json: current_user, status: :ok
    else
      render json: { errors: current_user.errors }, status: :unprocessable_entity
    end
  end

  def destroy
    current_user.destroy
    render json: {}, status: :no_content
  end

  private

  def user_params
    params.require(:user).permit(:email, :password)
  end
end
